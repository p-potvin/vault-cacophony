# commands

```
ffmpeg -i ".mp4" -af loudnorm=I=-16:TP=-1.5:LRA=11 -b:a 320k ".wav"
```

```
.\audiocpp_cli --task sep --family mel_band_roformer --model models/mel-roformer-mlx --backend cuda --audio "F:\amd\Models\newer pvt 2026-08-27\audio_temp.wav" --out-dir "F:\amd\Models\newer pvt 2026-08-27\mel_band" --session-option mel_band_roformer.num_overlap=1 --metrics
```
```
.\audiocpp_cli --task vad --family marblenet_vad --model assets/framework/models/marblenet_vad --backend cuda --audio "F:\amd\Models\newer pvt 2026-08-27\mel_band\vocals.wav" --segments-out "F:\amd\Models\newer pvt 2026-08-27\speech_timestamps.json" --request-option threshold=0.7 --metrics
```
```
.\audiocpp_cli --task asr --family nemotron_asr --model models/nemotron-3.5-asr-streaming-0.6b --backend cuda --audio "F:\amd\Models\newer pvt 2026-08-27\mel_band\vocals.wav" --language es-US --text-out "F:\amd\Models\newer pvt 2026-08-27\trancription.srt" --metrics
```