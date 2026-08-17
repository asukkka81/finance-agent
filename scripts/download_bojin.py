#!/usr/bin/env python3
"""下载博金数据集的核心文件."""
import urllib.request, ssl, json, os

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

os.makedirs("data/bojin", exist_ok=True)

# 1. 先看 dataset/ 目录
url = "https://modelscope.cn/api/v1/datasets/BJQW14B/bs_challenge_financial_14b_dataset/repo/tree?Source=SDK&Revision=master&Path=dataset"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
resp = urllib.request.urlopen(req, timeout=30, context=ctx)
data = json.loads(resp.read())

print("dataset/ files:")
for f in data["Data"]["Files"]:
    size_mb = f["Size"] / 1024 / 1024
    lfs = " [LFS]" if f.get("IsLFS") else ""
    print(f"  {f['Name']} ({size_mb:.1f} MB){lfs}")

# 2. 下载 question.json
base = "https://modelscope.cn/api/v1/datasets/BJQW14B/bs_challenge_financial_14b_dataset/repo?Source=SDK&Revision=master&FilePath=dataset%2F"

# question.json (already downloaded earlier, skip if exists)
qpath = "data/bojin/question.json"
if not os.path.exists(qpath):
    url2 = base + "question.json"
    print(f"\nDownloading question.json...")
    req = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=60, context=ctx)
    with open(qpath, "wb") as f:
        f.write(resp.read())
    print(f"  saved: {os.path.getsize(qpath)} bytes")
else:
    print(f"\nquestion.json already exists ({os.path.getsize(qpath)} bytes)")

# 3. 下载数据库 (1.46GB) — 尝试 URL 编码中文名
db_filename = "%E5%8D%9A%E9%87%91%E6%9D%AF%E6%AF%94%E8%B5%9B%E6%95%B0%E6%8D%AE.db"
dbpath = "data/bojin/bojin_finance.db"

if not os.path.exists(dbpath):
    url2 = base + db_filename
    print(f"\nDownloading database (1.46GB)...")
    req = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=600, context=ctx)
        total = int(resp.headers.get("Content-Length", 0))
        print(f"  size: {total/1024/1024:.1f} MB")
        downloaded = 0
        with open(dbpath, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded * 100 / total if total else 0
                print(f"  progress: {downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB ({pct:.0f}%)", end="\r")
        actual = os.path.getsize(dbpath)
        print(f"\n  saved: {actual/1024/1024:.1f} MB -> {dbpath}")
    except Exception as e:
        print(f"  failed: {e}")
else:
    print(f"\ndatabase already exists ({os.path.getsize(dbpath)/1024/1024:.1f} MB)")
