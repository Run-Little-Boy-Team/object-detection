conda create -y -n rlb
conda activate rlb
conda install -y python=3.10
conda install -y pytorch==2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install onnx onnxscript onnxruntime-gpu onnxsim opencv-python tqdm pycocotools