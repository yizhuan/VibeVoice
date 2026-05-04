git clone https://github.com/vibevoice-community/VibeVoice.git

uv run demo/inference_from_file.py \
  --model_path microsoft/VibeVoice-1.5B \
  --txt_path demo/text_examples/my_voice.txt \
  --speaker_names my_voice \
  --output_dir ./outputs


------------------
https://github.com/microsoft/VibeVoice
------------------


uv run demo/create_voice_preset.py \
    --audio_path demo/voices/news.wav \
    --transcript "中国没有窃取我们的繁荣，是我们的精英主动交出的。他们推动中国加入世界贸易组织，外包工厂，解散工会，并告诉美国工人学习编程。钢铁城镇衰落了，纺织厂消失了，汽车零部件产业被掏空了，但道琼斯指数创下历史新高，这才是他们唯一关心的。" \
    --output_path demo/voices/streaming_model/news.pt

uv run demo/realtime_model_inference_from_file.py \
    --speaker_name news \
    --txt_path demo/text_examples/1p_vibevoice.txt \
    --output_dir ./outputs
	
---	

uv run demo/create_voice_preset.py \
    --audio_path demo/voices/news.wav \
    --output_path demo/voices/streaming_model/news.pt \
    --skip_inversion
	
	
uv run demo/create_voice_pt.py \
  --input_audio demo/voices/my_voice.wav \
  --output_pt demo/voices/my_voice.pt

----

uv run demo/inference_from_file.py \
  --model_path microsoft/VibeVoice-1.5B \
  --txt_path demo/text_examples/my_voice.txt \
  --speaker_names my_voice \
  --output_dir ./outputs
	
	
	
	
	
uv run demo/realtime_model_inference_from_file.py \
    --speaker_name alex \
    --txt_path demo/text_examples/1p_vibevoice.txt \
    --output_dir ./outputs
	
	
	
git clone https://github.com/vibevoice-community/VibeVoice.git
cd VibeVoice
uv pip install -e .

python demo/inference_from_file.py \
  --model_path microsoft/VibeVoice-1.5B \
  --txt_path demo/text_examples/1p_abs.txt \
  --speaker_names Alice
  