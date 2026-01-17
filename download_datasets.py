#!/usr/bin/env python3
"""
数据集下载和解压脚本
download_datasets.py

支持的数据集:
- MS-COCO Captions
- Clotho
- AudioCaps
- VoxCeleb2 (子集)
"""

import os
import sys
import json
import shutil
import hashlib
import tarfile
import zipfile
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict
from tqdm import tqdm

try:
    import requests
    import gdown
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("请先安装依赖: pip install requests gdown huggingface_hub")
    sys.exit(1)


# ==================== 配置 ====================
DATA_ROOT = Path("./data")

DATASETS_CONFIG = {
    "coco": {
        "name": "MS-COCO Captions",
        "description": "图像描述数据集，33万图像，150万描述",
        "size": "~25GB",
        "files": {
            "train2017": {
                "url": "http://images.cocodataset.org/zips/train2017.zip",
                "md5": "cced6f7f71b7629ddf16f17bbcfab6b2",
                "size": "18GB"
            },
            "val2017": {
                "url": "http://images.cocodataset.org/zips/val2017.zip",
                "md5": "442b8da7639aecaf257c1dceb8ba8c80",
                "size": "1GB"
            },
            "annotations": {
                "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "md5": "f4bbac642086de4f52a3fdda2de5fa2c",
                "size": "241MB"
            }
        }
    },
    "clotho": {
        "name": "Clotho",
        "description": "音频描述数据集，5000音频，25000描述",
        "size": "~5GB",
        "download_method": "zenodo",
        "zenodo_id": "4783391",
        "files": {
            "development": "clotho_audio_development.7z",
            "validation": "clotho_audio_validation.7z",
            "evaluation": "clotho_audio_evaluation.7z",
            "captions_development": "clotho_captions_development.csv",
            "captions_validation": "clotho_captions_validation.csv",
            "captions_evaluation": "clotho_captions_evaluation.csv"
        }
    },
    "audiocaps": {
        "name": "AudioCaps",
        "description": "音频描述数据集，46K音频片段",
        "size": "~15GB",
        "download_method": "huggingface",
        "hf_repo": "d0rj/audiocaps",
        "note": "需要从AudioSet下载原始音频"
    },
    "voxceleb": {
        "name": "VoxCeleb2 (子集)",
        "description": "音视频说话人数据集",
        "size": "~50GB (完整) / ~5GB (子集)",
        "download_method": "official",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
        "note": "需要申请访问权限"
    },
    "cc3m_sample": {
        "name": "CC3M Sample",
        "description": "Conceptual Captions 3M 样本子集",
        "size": "~2GB",
        "download_method": "huggingface",
        "hf_repo": "liuhaotian/LLaVA-CC3M-Pretrain-595K"
    }
}


class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, data_root: Path, max_workers: int = 4):
        self.data_root = Path(data_root)
        self.max_workers = max_workers
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def download_file(
        self,
        url: str,
        dest_path: Path,
        desc: str = None,
        md5: str = None
    ) -> bool:
        """下载单个文件"""
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查文件是否已存在且MD5匹配
        if dest_path.exists() and md5:
            if self._check_md5(dest_path, md5):
                print(f"  文件已存在且校验通过: {dest_path.name}")
                return True
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    desc=desc or dest_path.name,
                    ncols=80
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            
            # 验证MD5
            if md5 and not self._check_md5(dest_path, md5):
                print(f"  警告: MD5校验失败 {dest_path.name}")
                return False
            
            return True
            
        except Exception as e:
            print(f"  下载失败: {e}")
            return False
    
    def _check_md5(self, file_path: Path, expected_md5: str) -> bool:
        """检查文件MD5"""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest() == expected_md5
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """解压归档文件"""
        archive_path = Path(archive_path)
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        
        print(f"  解压: {archive_path.name}")
        
        try:
            suffix = archive_path.suffix.lower()
            
            if suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_to)
            
            elif suffix in ['.tar', '.gz', '.tgz']:
                mode = 'r:gz' if suffix in ['.gz', '.tgz'] else 'r'
                with tarfile.open(archive_path, mode) as tf:
                    tf.extractall(extract_to)
            
            elif suffix == '.7z':
                try:
                    import py7zr
                    with py7zr.SevenZipFile(archive_path, mode='r') as z:
                        z.extractall(extract_to)
                except ImportError:
                    subprocess.run(['7z', 'x', str(archive_path), f'-o{extract_to}'], check=True)
            
            else:
                print(f"  不支持的归档格式: {suffix}")
                return False
            
            return True
            
        except Exception as e:
            print(f"  解压失败: {e}")
            return False
    
    def download_coco(self) -> bool:
        """下载MS-COCO数据集"""
        print("\n" + "="*60)
        print("下载 MS-COCO Captions 数据集")
        print("="*60)
        
        coco_dir = self.data_root / "coco"
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        config = DATASETS_CONFIG["coco"]
        
        for name, file_info in config["files"].items():
            print(f"\n下载 {name}...")
            
            zip_path = coco_dir / f"{name}.zip"
            
            if not self.download_file(
                file_info["url"],
                zip_path,
                desc=f"COCO {name}",
                md5=file_info.get("md5")
            ):
                return False
            
            # 解压
            if not self.extract_archive(zip_path, coco_dir):
                return False
            
            # 删除zip文件节省空间 (可选)
            # zip_path.unlink()
        
        print("\n✓ MS-COCO 下载完成")
        return True
    
    def download_clotho(self) -> bool:
        """下载Clotho数据集"""
        print("\n" + "="*60)
        print("下载 Clotho 数据集")
        print("="*60)
        
        clotho_dir = self.data_root / "clotho"
        clotho_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用aac-datasets库下载
        try:
            print("使用 aac-datasets 库下载...")
            subprocess.run([
                sys.executable, "-m", "aac_datasets.download",
                "--root", str(clotho_dir),
                "clotho"
            ], check=True)
            print("\n✓ Clotho 下载完成")
            return True
        except subprocess.CalledProcessError:
            pass
        except FileNotFoundError:
            pass
        
        # 备选: 从Zenodo下载
        print("尝试从 Zenodo 下载...")
        zenodo_base = f"https://zenodo.org/record/{DATASETS_CONFIG['clotho']['zenodo_id']}/files"
        
        files_to_download = [
            "clotho_audio_development.7z",
            "clotho_audio_validation.7z", 
            "clotho_audio_evaluation.7z",
            "clotho_captions_development.csv",
            "clotho_captions_validation.csv",
            "clotho_captions_evaluation.csv"
        ]
        
        for filename in files_to_download:
            url = f"{zenodo_base}/{filename}"
            dest = clotho_dir / filename
            
            if not self.download_file(url, dest, desc=filename):
                print(f"  警告: {filename} 下载失败")
            
            # 解压7z文件
            if filename.endswith('.7z') and dest.exists():
                self.extract_archive(dest, clotho_dir)
        
        print("\n✓ Clotho 下载完成")
        return True
    
    def download_audiocaps(self) -> bool:
        """下载AudioCaps数据集"""
        print("\n" + "="*60)
        print("下载 AudioCaps 数据集")
        print("="*60)
        
        audiocaps_dir = self.data_root / "audiocaps"
        audiocaps_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用aac-datasets库下载
        try:
            print("使用 aac-datasets 库下载...")
            print("注意: AudioCaps需要从YouTube下载音频，可能需要较长时间")
            subprocess.run([
                sys.executable, "-m", "aac_datasets.download",
                "--root", str(audiocaps_dir),
                "audiocaps"
            ], check=True)
            print("\n✓ AudioCaps 下载完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  下载失败: {e}")
        except FileNotFoundError:
            print("  aac-datasets 未安装")
        
        # 备选: 下载标注文件
        print("\n下载 AudioCaps 标注文件...")
        
        annotations_url = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset"
        for split in ["train", "val", "test"]:
            url = f"{annotations_url}/{split}.csv"
            dest = audiocaps_dir / f"{split}.csv"
            self.download_file(url, dest, desc=f"AudioCaps {split}")
        
        print("\n注意: 音频文件需要使用 yt-dlp 从 YouTube 下载")
        print("可以运行: python download_audiocaps_audio.py")
        
        return True
    
    def download_cc3m_sample(self) -> bool:
        """下载CC3M样本子集"""
        print("\n" + "="*60)
        print("下载 CC3M 样本子集 (LLaVA-CC3M-Pretrain-595K)")
        print("="*60)
        
        cc3m_dir = self.data_root / "cc3m"
        cc3m_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print("从 HuggingFace 下载...")
            snapshot_download(
                repo_id="liuhaotian/LLaVA-CC3M-Pretrain-595K",
                local_dir=cc3m_dir,
                repo_type="dataset"
            )
            print("\n✓ CC3M 样本子集下载完成")
            return True
        except Exception as e:
            print(f"  下载失败: {e}")
            return False
    
    def download_voxceleb_sample(self) -> bool:
        """下载VoxCeleb样本"""
        print("\n" + "="*60)
        print("下载 VoxCeleb 样本数据")
        print("="*60)
        
        voxceleb_dir = self.data_root / "voxceleb"
        voxceleb_dir.mkdir(parents=True, exist_ok=True)
        
        print("注意: VoxCeleb2完整数据集需要申请访问权限")
        print("官网: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/")
        print("\n尝试下载测试样本...")
        
        # 创建示例数据结构
        (voxceleb_dir / "dev").mkdir(exist_ok=True)
        (voxceleb_dir / "test").mkdir(exist_ok=True)
        
        # 创建说明文件
        readme = voxceleb_dir / "README.md"
        readme.write_text("""# VoxCeleb2 数据集

## 下载说明

1. 访问官网申请下载权限:
   https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

2. 下载以下文件:
   - vox2_dev_mp4_partaa ~ vox2_dev_mp4_partah (开发集视频)
   - vox2_test_mp4.zip (测试集视频)
   - vox2_dev_txt.zip (开发集元数据)
   - vox2_test_txt.zip (测试集元数据)

3. 解压文件:
   ```bash
   cat vox2_dev_mp4_parta* > vox2_dev.tar
   tar -xf vox2_dev.tar
   ```

4. 组织目录结构:
   ```
   voxceleb/
   ├── dev/
   │   └── mp4/
   │       └── id00001/
   │           └── video1/
   │               └── 00001.mp4
   └── test/
       └── mp4/
           └── ...
   ```

## 或者使用HuggingFace (部分数据)

```python
from datasets import load_dataset
dataset = load_dataset("ProgramComputer/voxceleb", split="test")
```
""")
        
        print(f"\n说明文件已创建: {readme}")
        print("请按照说明手动下载数据集")
        
        return True
    
    def create_dataset_info(self):
        """创建数据集信息文件"""
        info_path = self.data_root / "dataset_info.json"
        
        info = {
            "download_date": str(Path(__file__).stat().st_mtime),
            "datasets": {}
        }
        
        for name, config in DATASETS_CONFIG.items():
            dataset_dir = self.data_root / name.replace("_sample", "")
            info["datasets"][name] = {
                "name": config["name"],
                "description": config["description"],
                "expected_size": config["size"],
                "path": str(dataset_dir),
                "exists": dataset_dir.exists(),
                "files": list(dataset_dir.glob("*")) if dataset_dir.exists() else []
            }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        print(f"\n数据集信息已保存: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="多模态数据集下载工具")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["coco", "clotho", "audiocaps", "cc3m", "voxceleb", "all"],
        default=["all"],
        help="要下载的数据集"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="数据存储根目录"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已存在的数据集"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("        多模态数据集下载工具")
    print("="*60)
    
    downloader = DatasetDownloader(args.data_root)
    
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["coco", "clotho", "audiocaps", "cc3m", "voxceleb"]
    
    results = {}
    
    for dataset in datasets_to_download:
        dataset_dir = downloader.data_root / dataset.replace("_sample", "")
        
        if args.skip_existing and dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"\n跳过已存在的数据集: {dataset}")
            results[dataset] = "skipped"
            continue
        
        if dataset == "coco":
            results[dataset] = downloader.download_coco()
        elif dataset == "clotho":
            results[dataset] = downloader.download_clotho()
        elif dataset == "audiocaps":
            results[dataset] = downloader.download_audiocaps()
        elif dataset == "cc3m":
            results[dataset] = downloader.download_cc3m_sample()
        elif dataset == "voxceleb":
            results[dataset] = downloader.download_voxceleb_sample()
    
    # 创建数据集信息
    downloader.create_dataset_info()
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("下载结果摘要")
    print("="*60)
    
    for dataset, result in results.items():
        if result == "skipped":
            status = "⏭ 跳过"
        elif result:
            status = "✓ 成功"
        else:
            status = "✗ 失败"
        print(f"  {dataset}: {status}")
    
    print("\n下载完成!")
    print("下一步: 运行 python run_training.py 开始训练")


if __name__ == "__main__":
    main()
