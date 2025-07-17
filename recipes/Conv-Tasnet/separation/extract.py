#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torchaudio
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger


class Separation(sb.Brain):
    def compute_forward(self, mix, stage):
        mix, _ = mix
        mix = mix.to(self.device)
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask
        est_source = torch.cat(
            [self.hparams.Decoder(sep_h[i]).unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        )
        t_origin = mix.size(1)
        t_est = est_source.size(1)
        if t_origin > t_est:
            est_source = torch.nn.functional.pad(est_source, (0, 0, 0, t_origin - t_est))
        else:
            est_source = est_source[:, :t_origin, :]
        return est_source

    def save_audio(self, snt_id, mixture, predictions):
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        os.makedirs(save_path, exist_ok=True)
        for ns in range(self.hparams.num_spks):
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            torchaudio.save(
                os.path.join(save_path, f"{snt_id}_source{ns+1}hat.wav"),
                signal.unsqueeze(0).cpu(),
                self.hparams.sample_rate,
            )
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        torchaudio.save(
            os.path.join(save_path, f"{snt_id}_mix.wav"),
            signal.unsqueeze(0).cpu(),
            self.hparams.sample_rate,
        )


def dataio_prep(hparams):
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        return sb.dataio.dataio.read_audio(mix_wav)

    sb.dataio.dataset.add_dynamic_item([test_data], audio_pipeline_mix)
    sb.dataio.dataset.set_output_keys([test_data], ["id", "mix_sig"])
    return test_data


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    sb.utils.distributed.ddp_init_group(run_opts)
    get_logger(__name__)
    test_data = dataio_prep(hparams)
    if "pretrained_separator" in hparams:
        sb.utils.distributed.run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()
    separator = Separation(
        modules=hparams["modules"],
        opt_class=None,
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    test_loader = sb.dataio.dataloader.make_dataloader(
        test_data, **hparams["dataloader_opts"]
    )

    with torch.no_grad():
        for batch in tqdm(test_loader):
            snt_id = batch.id[0]
            mixture = batch.mix_sig
            preds = separator.compute_forward(mixture, sb.Stage.TEST)
            separator.save_audio(snt_id, mixture, preds)
