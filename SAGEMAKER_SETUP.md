# SageMaker Image Generation Setup Guide

## What You Have Now

✅ **requirements.txt** - All Python dependencies exported from your uv setup
✅ **launch_sagemaker_generation.py** - SageMaker launcher script
✅ **AWS credentials** - Configured via SSO (profile: Timothe)
✅ **S3 bucket** - stylegan2-generated-images_v2
✅ **IAM Role** - arn:aws:iam::183295429625:role/service-role/AmazonSageMaker-ExecutionRole-20241001T141781

## How It Works

1. **ScriptProcessor** uploads your entire repo to SageMaker
2. Installs dependencies from requirements.txt
3. Runs generate.py on GPU instances (ml.g4dn.xlarge)
4. Each instance generates a portion of the 1M images
5. Images are continuously uploaded to S3 as they're generated

## Step-by-Step Instructions

### Step 1: Install SageMaker Python SDK

```bash
uv pip install sagemaker boto3
```

### Step 2: (Optional) Upload Your Trained Model to S3

If you have a custom trained model (not using NVIDIA's pretrained):

```bash
# Upload pre-trained VGG checkpoint to S3
aws s3 cp /home/sagemaker-user/stylegan2-ada-pytorch/dex_age_classifier.pth \
s3://emobot-prod-workspace-bucket/usecases/emobot-research/datasets/stylegan2-generated-images_v2/models/vgg.pth

# Upload weight vector
aws s3 cp /home/sagemaker-user/stylegan2-ada-pytorch/weight.npy \
s3://emobot-prod-workspace-bucket/usecases/emobot-research/datasets/stylegan2-generated-images_v2/models/weight.npy
```

### Step 3: Run a Test Job (RECOMMENDED)

Start with a small test to verify everything works:

```bash
python launch_sagemaker_generation.py --test
```

This will:
- Generate only 100 images
- Use 1 instance
- Run for max 1 hour
- Help verify your setup works

### Step 4: Launch Full Production Job

Once the test succeeds, launch the full job:

```bash
# Using NVIDIA's pretrained FFHQ model (no upload needed)
python launch_sagemaker_generation.py \
    --total-images 1000000 \
    --instances 10 \
    --role arn:aws:iam::183295429625:role/service-role/AmazonSageMaker-ExecutionRole-20241001T141781

# OR with your custom model
python launch_sagemaker_generation.py \
    --total-images 1000000 \
    --instances 10 \
    --role arn:aws:iam::183295429625:role/service-role/AmazonSageMaker-ExecutionRole-20241001T141781 \
    --model-s3 s3://stylegan2-generated-images_v2/models/model.pkl

# OR with weight vector modulation
python launch_sagemaker_generation.py \
    --total-images 1000000 \
    --instances 10 \
    --role arn:aws:iam::183295429625:role/service-role/AmazonSageMaker-ExecutionRole-20241001T141781 \
    --weight-vector-s3 s3://stylegan2-generated-images_v2/models/weight.npy
```

### Step 5: Monitor Progress

Monitor your job at:
https://console.aws.amazon.com/sagemaker/home?region=eu-west-3#/processing-jobs

Or via CLI:
```bash
aws sagemaker list-processing-jobs --profile Timothe
```

### Step 6: Access Generated Images

Images will be uploaded to:
```
s3://stylegan2-generated-images_v2/generated_images_YYYYMMDD_HHMMSS/
```

Download them:
```bash
aws s3 sync s3://stylegan2-generated-images_v2/generated_images_YYYYMMDD_HHMMSS/ ./downloaded_images/ --profile Timothe
```

## Configuration Options

### Instance Types

- **ml.g4dn.xlarge** (1 GPU, 16 GB RAM) - ~$0.736/hour - Default
- **ml.g4dn.2xlarge** (1 GPU, 32 GB RAM) - ~$0.94/hour - More memory
- **ml.g4dn.4xlarge** (1 GPU, 64 GB RAM) - ~$1.50/hour - If you need more RAM

### Cost Estimation

For 1M images with default settings:
- 10 instances × ml.g4dn.xlarge × 24 hours = ~$176
- Adjust instance_count and max_runtime based on your needs

### Parallelization Strategy

The script splits images evenly across instances:
- **10 instances** → 100K images each
- **20 instances** → 50K images each (faster but more expensive)
- **5 instances** → 200K images each (cheaper but slower)

## Customizing Generation

Edit `launch_sagemaker_generation.py` to customize:

```python
launch_image_generation(
    total_images=1_000_000,
    instance_count=10,
    truncation_psi=0.7,        # Adjust truncation
    noise_mode='random',       # or 'const', 'none'
    alphas="-5,0,5",          # For weight modulation
    style_range=(0, 17),      # Style block range
    max_runtime_hours=24,
)
```

## Troubleshooting

### Issue: "Could not auto-detect SageMaker role"
**Solution:** Pass the role explicitly with `--role` flag

### Issue: "Access Denied" when uploading to S3
**Solution:** Check your IAM role has S3 write permissions:
```bash
aws iam get-role-policy --role-name AmazonSageMaker-ExecutionRole-20241001T141781 --policy-name S3Access --profile Timothe
```

### Issue: Job fails with "CUDA out of memory"
**Solution:** Reduce batch size or use larger instance (ml.g4dn.2xlarge)

### Issue: Dependencies fail to install
**Solution:** The requirements.txt is comprehensive. If issues persist, you may need to create a custom Docker image.

## Next Steps After Setup

1. ✅ Run test job with `--test` flag
2. ✅ Verify images appear in S3
3. ✅ Launch full production job
4. ✅ Monitor progress in SageMaker console
5. ✅ Download and use generated images

## Important Notes

- **Continuous Upload**: Images upload to S3 as they're generated (no need to wait for job completion)
- **Resumability**: If a job fails, you can restart with different seed ranges
- **Cost Control**: Set `max_runtime_hours` to prevent runaway costs
- **Region**: All resources are in eu-west-3 (Paris)
