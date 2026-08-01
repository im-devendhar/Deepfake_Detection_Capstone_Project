# AWS EC2 Deployment

This project is a Django + PyTorch deepfake detection app. The simplest AWS deployment is an EC2 instance running Docker.

## 1. Launch EC2

- AMI: Ubuntu Server 22.04 LTS or 24.04 LTS
- Instance type: `t3.large` recommended for CPU inference
- Storage: at least 20 GB
- Security group inbound rules:
  - SSH: port 22 from your IP
  - HTTP: port 80 from anywhere

## 2. Install Docker On EC2

```bash
sudo apt update
sudo apt install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME}) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ubuntu
```

Log out and SSH back in after installing Docker.

## 3. Clone The Repository

```bash
git clone https://github.com/im-devendhar/Deepfake_Detection_Capstone_Project.git
cd Deepfake_Detection_Capstone_Project/Django\ Application
```

## 4. Add Model Files

The trained `.pt` model files are not committed to GitHub because they are large. Copy them into:

```text
Django Application/models/
```

Required files:

```text
model_87_acc_20_frames_final_data.pt
model_89_acc_40_frames_final_data.pt
model_90_acc_60_frames_final_data.pt
model_97_acc_80_frames_FF_data.pt
model_97_acc_100_frames_FF_data.pt
```

You can upload them from your computer with `scp`:

```bash
scp -i your-key.pem "Django Application/models/*.pt" ubuntu@YOUR_EC2_PUBLIC_IP:/home/ubuntu/Deepfake_Detection_Capstone_Project/Django\ Application/models/
```

## 5. Configure Environment

```bash
cp .env.aws.example .env
nano .env
```

Set:

```text
DJANGO_SECRET_KEY=your-long-secret-key
DJANGO_ALLOWED_HOSTS=YOUR_EC2_PUBLIC_IP
```

## 6. Build And Run

```bash
docker build -f Dockerfile.aws -t deepfake-detection .
docker run -d --name deepfake-detection --env-file .env -p 80:8000 deepfake-detection
```

Open:

```text
http://YOUR_EC2_PUBLIC_IP
```

## Useful Commands

```bash
docker logs -f deepfake-detection
docker restart deepfake-detection
docker stop deepfake-detection
docker rm deepfake-detection
```

## Re-deploy After Code Changes

```bash
git pull
docker stop deepfake-detection
docker rm deepfake-detection
docker build -f Dockerfile.aws -t deepfake-detection .
docker run -d --name deepfake-detection --env-file .env -p 80:8000 deepfake-detection
```
