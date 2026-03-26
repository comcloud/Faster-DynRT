import json
import pickle
import torch
import os
import numpy as np
from PIL import Image
try:
    import torchvision.transforms as transforms
except Exception:  # noqa: BLE001
    transforms = None


def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

def load_2_file(filename):
    id_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
        for data in data_list:
            id_list.append(data['image_id'])
    return id_list
def load_bully_file(filename):

    # 读取文件并将每行作为列表元素
    with open(filename, 'r') as file:
        data_list = file.readlines()

    # 去除每行末尾的换行符（如果需要）
    return [line.strip() for line in data_list]

class loader_img:
    def __init__(self):
        self.name="img"
        self.require=[]

    def prepare(self,input,opt):
        source = opt.get("source", "bully")
        if source == "prepared":
            self.id = {
                "train": load_file(opt["data_path"] + "train_id"),
                "test": load_file(opt["data_path"] + "test_id"),
                "valid": load_file(opt["data_path"] + "valid_id")
            }
        elif source == "msd2":
            self.id = {
                "train": load_2_file(opt["data_2_path"] + "train.json"),
                "test": load_2_file(opt["data_2_path"] + "test.json"),
                "valid": load_2_file(opt["data_2_path"] + "valid.json")
            }
        elif source == "bully":
            self.id = {
                "train": load_bully_file(opt["data_bully_path"] + "train_id.txt"),
                "test": load_bully_file(opt["data_bully_path"] + "test_id.txt"),
                "valid": load_bully_file(opt["data_bully_path"] + "valid_id.txt")
            }
        else:
            raise ValueError(f"Unsupported image loader source: {source}")

        self.transform_image_path = opt["transform_image"]
        self.image_root = opt.get("image_root")
        self.image_resize = opt.get("image_resize", 224)
        if self.transform_image_path and not os.path.exists(self.transform_image_path):
            os.makedirs(self.transform_image_path, exist_ok=True)

    # def get(self,result,mode,index):
    #     img_path=os.path.join(
    #             self.transform_image_path,
    #             "{}.npy".format(self.id[mode][index])
    #         )
    #     img = torch.from_numpy(np.load(img_path))
    #     result["img"]=img


    def getlength(self,mode):
        return len(self.id[mode])

    def _resolve_tensor_path(self, sample_id):
        sid = str(sample_id)
        stem = os.path.splitext(sid)[0]
        candidates = [
            os.path.join(self.transform_image_path, f"{sid}.npy"),
            os.path.join(self.transform_image_path, f"{stem}.npy")
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[-1]

    def _resolve_image_path(self, sample_id):
        if not self.image_root:
            return None
        sid = str(sample_id)
        stem = os.path.splitext(sid)[0]
        ext = os.path.splitext(sid)[1]
        candidates = [os.path.join(self.image_root, sid)]
        if ext:
            candidates.extend([
                os.path.join(self.image_root, stem + ".jpg"),
                os.path.join(self.image_root, stem + ".png"),
                os.path.join(self.image_root, stem + ".jpeg")
            ])
        else:
            candidates.extend([
                os.path.join(self.image_root, stem + ".jpg"),
                os.path.join(self.image_root, stem + ".png"),
                os.path.join(self.image_root, stem + ".jpeg")
            ])
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def get(self, result, mode, index):
        sample_id = self.id[mode][index]
        tensor_path = self._resolve_tensor_path(sample_id)
        if not os.path.exists(tensor_path):
            image_path = self._resolve_image_path(sample_id)
            if image_path is None:
                raise FileNotFoundError(
                    f"Missing image tensor for '{sample_id}' at '{tensor_path}', and source image not found. "
                    f"Set dataloader.loaders.img.image_root correctly."
                )
            if transforms is None:
                raise RuntimeError("torchvision is required to transform raw images into tensor npy files")
            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((self.image_resize, self.image_resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = transform(image)
            np.save(tensor_path, image.numpy())
        img = torch.from_numpy(np.load(tensor_path))
        result["img"] = img

    def add_salt_and_pepper_noise(self, X, noise_factor):
        row, col, _ = X.shape
        num_salt = int(noise_factor * row * col)
        num_pepper = int(noise_factor * row * col)

        # 盐噪声（白点）
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in X.shape]
        X[salt_coords[0], salt_coords[1], :] = 1  # 设置为最大值

        # 椒噪声（黑点）
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in X.shape]
        X[pepper_coords[0], pepper_coords[1], :] = 0  # 设置为最小值

        return X

    def add_poisson_noise(self, X, noise_factor):
        # 为每个像素生成泊松噪声
        noisy = np.random.poisson(X * noise_factor) / noise_factor
        return noisy

    def add_speckle_noise(self, X, noise_factor):
        row, col, _ = X.shape
        gauss = np.random.normal(0, noise_factor, (row, col, X.shape[2]))  # 生成高斯噪声
        noisy = X + X * gauss  # 噪声与原图像值成比例
        return noisy

    def add_uniform_noise(self, X, noise_factor):
        row, col, _ = X.shape
        noise = np.random.uniform(-noise_factor, noise_factor, X.shape)  # 在[-noise_factor, noise_factor]范围内生成噪声
        X_noisy = X + noise
        return X_noisy

    def add_gaussian_noise(self, X, noise_factor):
        noise = np.random.normal(0, noise_factor, X.shape)
        X_noisy = X + noise
        return X_noisy
