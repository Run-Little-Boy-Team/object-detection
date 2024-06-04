conda create -y -n rlb
conda activate rlb
conda install -y python=3.10
conda install -y pytorch==2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install onnx onnxscript onnxsim opencv-python tqdm pycocotools

sudo apt install -y build-essential gcc g++ libprotobuf-dev protobuf-compiler libomp-dev libvulkan-dev

name="ncnn"
url="https://github.com/Tencent/ncnn"
echo "Cloning $url into $name"
git clone $url $name
cd $name
git submodule update --init
echo "Compiling $name"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_REQUANT=ON -DNCNN_BUILD_EXAMPLES=OFF ..
make -j4
echo "Installing $name"
make install
sudo cp -rf ./install/* /usr/local/
cd ../..
rm -rf $name