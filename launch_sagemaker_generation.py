#!/usr/bin/env python3
"""
Launch SageMaker ScriptProcessor to generate images using generate.py
"""
import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from datetime import datetime


def launch_image_generation(
    # Generation parameters
    total_images=1_000_000,
    instance_count=10,
    network_pkl_s3=None,  # S3 path to your model, or None to use NVIDIA's pretrained
    weight_vector_s3=None,  # S3 path to weight.npy if needed

    # SageMaker configuration
    role=None,  # Will auto-detect if None
    instance_type='ml.g4dn.xlarge',  # GPU instance
    max_runtime_hours=24,

    # Generation settings
    truncation_psi=0.7,
    noise_mode='random',
    alphas=None,  # e.g., "-5:-5:1" or None for no weight modulation
    style_range=(0, 17),
):
    """
    Launch SageMaker processing job to generate images.

    Args:
        total_images: Total number of images to generate
        instance_count: Number of parallel instances
        network_pkl_s3: S3 path to model pickle (or None for NVIDIA pretrained)
        weight_vector_s3: S3 path to weight vector .npy file
        role: SageMaker execution role ARN (auto-detected if None)
        instance_type: EC2 instance type (ml.g4dn.xlarge = 1 GPU)
        max_runtime_hours: Maximum job runtime in hours
        truncation_psi: Truncation parameter for generation
        noise_mode: 'random', 'const', or 'none'
        alphas: Alpha values for weight modulation (e.g., "-5,0,5")
        style_range: Tuple of (start, end) for style block range
    """

    # Initialize session
    session = sagemaker.Session()
    region = session.boto_region_name
    bucket = 'stylegan2-generated-images_v2'

    # Get execution role
    if role is None:
        try:
            role = sagemaker.get_execution_role()
            print(f"✓ Using execution role: {role}")
        except:
            raise ValueError(
                "Could not auto-detect SageMaker role. Please provide role ARN.\n"
                "Create a role at: https://console.aws.amazon.com/iam/home#/roles\n"
                "Or get existing role with: aws iam list-roles --query 'Roles[?contains(RoleName, `SageMaker`)].Arn'"
            )

    # Calculate seeds per instance
    images_per_instance = total_images // instance_count
    print(f"\n{'='*60}")
    print(f"SageMaker Image Generation Configuration")
    print(f"{'='*60}")
    print(f"Total images:          {total_images:,}")
    print(f"Instance count:        {instance_count}")
    print(f"Images per instance:   {images_per_instance:,}")
    print(f"Instance type:         {instance_type}")
    print(f"Output S3 bucket:      s3://{bucket}/")
    print(f"Max runtime:           {max_runtime_hours} hours")
    print(f"Region:                {region}")
    print(f"{'='*60}\n")

    # Set default network pickle if not provided
    if network_pkl_s3 is None:
        network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
        print("ℹ️  Using NVIDIA pretrained FFHQ model")
    else:
        network_pkl = network_pkl_s3
        print(f"ℹ️  Using custom model: {network_pkl}")

    # Create ScriptProcessor
    print("\n📦 Creating ScriptProcessor...")
    processor = ScriptProcessor(
        image_uri=f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.0.0-gpu-py310',
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size_in_gb=100,
        max_runtime_in_seconds=max_runtime_hours * 3600,
        base_job_name='stylegan2-generation',
        sagemaker_session=session,
    )

    # Calculate seed ranges for each instance
    jobs = []
    for instance_idx in range(instance_count):
        start_seed = instance_idx * images_per_instance
        end_seed = start_seed + images_per_instance - 1

        # Build arguments for generate.py
        arguments = [
            '--network', network_pkl,
            '--seeds', f'{start_seed}-{end_seed}',
            '--trunc', str(truncation_psi),
            '--noise-mode', noise_mode,
            '--outdir', '/opt/ml/processing/output',
        ]

        # Add weight vector arguments if provided
        if weight_vector_s3 is not None:
            arguments.extend([
                '--weight-vector', '/opt/ml/processing/input/weight.npy',
            ])
            if alphas is not None:
                arguments.extend(['--alphas', alphas])
            arguments.extend(['--style-range', str(style_range[0]), str(style_range[1])])

        jobs.append({
            'instance_idx': instance_idx,
            'start_seed': start_seed,
            'end_seed': end_seed,
            'arguments': arguments
        })

    # Prepare inputs
    inputs = []
    if weight_vector_s3 is not None:
        inputs.append(
            ProcessingInput(
                source=weight_vector_s3,
                destination='/opt/ml/processing/input',
                input_name='weight_vector'
            )
        )

    # Prepare outputs
    outputs = [
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination=f's3://{bucket}/generated_images_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            s3_upload_mode='Continuous',  # Upload files as they're generated
            output_name='images'
        )
    ]

    # Display job configuration
    print(f"\n🚀 Launching {instance_count} processing jobs...")
    print(f"\nJob details:")
    for job in jobs[:3]:  # Show first 3
        print(f"  Instance {job['instance_idx']}: seeds {job['start_seed']:,} to {job['end_seed']:,}")
    if instance_count > 3:
        print(f"  ... and {instance_count - 3} more instances")

    # Launch the processing job
    print(f"\n⏳ Starting processing job...")
    print(f"💡 Tip: Monitor progress at: https://console.aws.amazon.com/sagemaker/home?region={region}#/processing-jobs")

    try:
        processor.run(
            code='generate.py',
            source_dir='.',  # Upload entire current directory
            dependencies='requirements.txt',  # Install dependencies
            inputs=inputs,
            outputs=outputs,
            arguments=jobs[0]['arguments'],  # All instances will run same args but different seeds
            wait=False,  # Don't wait for completion
            logs=True,
        )

        print(f"\n✅ Processing job launched successfully!")
        print(f"📊 Job name: {processor.latest_job.name}")
        print(f"🪣 Output location: {outputs[0].destination}")
        print(f"\nℹ️  Images will be continuously uploaded to S3 as they're generated.")
        print(f"ℹ️  Expected completion: ~{max_runtime_hours} hours")

        return processor

    except Exception as e:
        print(f"\n❌ Error launching job: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Launch SageMaker image generation')
    parser.add_argument('--total-images', type=int, default=1_000_000,
                        help='Total number of images to generate')
    parser.add_argument('--instances', type=int, default=10,
                        help='Number of parallel instances')
    parser.add_argument('--role', type=str, default=None,
                        help='SageMaker execution role ARN')
    parser.add_argument('--model-s3', type=str, default=None,
                        help='S3 path to model pickle (optional)')
    parser.add_argument('--weight-vector-s3', type=str, default=None,
                        help='S3 path to weight.npy (optional)')
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                        help='SageMaker instance type')
    parser.add_argument('--test', action='store_true',
                        help='Run a small test job (100 images, 1 instance)')

    args = parser.parse_args()

    # Test mode: generate only 100 images on 1 instance
    if args.test:
        print("🧪 TEST MODE: Generating 100 images on 1 instance")
        launch_image_generation(
            total_images=100,
            instance_count=1,
            role=args.role,
            network_pkl_s3=args.model_s3,
            weight_vector_s3=args.weight_vector_s3,
            instance_type=args.instance_type,
            max_runtime_hours=1,
        )
    else:
        # Production mode
        launch_image_generation(
            total_images=args.total_images,
            instance_count=args.instances,
            role=args.role,
            network_pkl_s3=args.model_s3,
            weight_vector_s3=args.weight_vector_s3,
            instance_type=args.instance_type,
        )
