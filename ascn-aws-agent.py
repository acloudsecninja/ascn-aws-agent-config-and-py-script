# Import required libraries
import os
import boto3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Fetch credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY')
aws_secret_access_key = os.getenv('AWS_SECRET_KEY')
region_name = os.getenv('REGION_NAME')

# Create AWS service clients
ec2 = boto3.client('ec2', aws_access_key_id=aws_access_key_id, 
                   aws_secret_access_key=aws_secret_access_key, region_name=region_name)
route53 = boto3.client('route53', aws_access_key_id=aws_access_key_id, 
                      aws_secret_access_key=aws_secret_access_key, region_name=region_name)
iam = boto3.client('iam', aws_access_key_id=aws_access_key_id, 
                   aws_secret_access_key=aws_secret_access_key, region_name=region_name)

@tool
def aws_cli_command(command: str) -> str:
    """Execute AWS CLI commands for interacting with AWS services"""
    import subprocess
    try:
        # Set AWS environment variables
        env = os.environ.copy()
        env['AWS_ACCESS_KEY_ID'] = aws_access_key_id
        env['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
        env['AWS_DEFAULT_REGION'] = region_name
        
        # Execute AWS CLI command
        result = subprocess.run(['aws'] + command.split(), 
                              capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error executing AWS command: {str(e)}"

@tool
def list_route53_hosted_zones() -> str:
    """List all Route 53 hosted zones in the AWS account"""
    try:
        response = route53.list_hosted_zones()
        hosted_zones = [zone['Name'] for zone in response['HostedZones']]
        return f"Route 53 Hosted Zones: {hosted_zones}"
    except Exception as e:
        return f"Error listing Route 53 hosted zones: {str(e)}"

@tool
def get_ec2_instance_size(instance_ip: str) -> str:
    """Get the instance type/size of an EC2 instance by its private IP address"""
    try:
        instances = ec2.describe_instances(Filters=[{'Name': 'private-ip-address', 'Values': [instance_ip]}])
        if instances['Reservations']:
            instance_type = instances['Reservations'][0]['Instances'][0]['InstanceType']
            return f"EC2 instance {instance_ip} is of type: {instance_type}"
        else:
            return f"No EC2 instance found with IP address: {instance_ip}"
    except Exception as e:
        return f"Error getting EC2 instance size: {str(e)}"

@tool
def get_user_permissions(user_name: str) -> str:
    """Get all attached IAM policies for a specific IAM user"""
    try:
        policies = iam.list_attached_user_policies(UserName=user_name)
        policy_names = [policy['PolicyName'] for policy in policies['AttachedPolicies']]
        return f"IAM user '{user_name}' has these policies: {policy_names}"
    except Exception as e:
        return f"Error getting user permissions: {str(e)}"

@tool
def list_s3_buckets() -> str:
    """List all S3 buckets in the AWS account"""
    try:
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, 
                        aws_secret_access_key=aws_secret_access_key, region_name=region_name)
        response = s3.list_buckets()
        bucket_names = [bucket['Name'] for bucket in response['Buckets']]
        return f"S3 Buckets: {bucket_names}"
    except Exception as e:
        return f"Error listing S3 buckets: {str(e)}"

# Initialize OpenAI chat model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that can interact with AWS services. You have access to the following tools:

Available tools:
- aws_cli_command: Execute any AWS CLI command
- list_route53_hosted_zones: List all Route 53 hosted zones
- get_ec2_instance_size: Get EC2 instance type by private IP address
- get_user_permissions: Get IAM user policies by username
- list_s3_buckets: List all S3 buckets

User request: {input}

Your response:
""")

# Create the chain
chain = prompt | llm

def test_aws_agent():
    """Test the AWS agent with various AWS operations"""
    test_queries = [
        "List all S3 buckets in my AWS account",
        "List all Route 53 hosted zones", 
        "Get the size of EC2 instance with IP 10.0.1.112",
        "Get permissions for IAM user take-home-coding"
    ]
    
    for query in test_queries:
        try:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print('='*50)
            
            # Get LLM response
            result = chain.invoke({"input": query})
            print("Agent response:")
            print(result.content)
            
            # Execute relevant tool based on query
            if "s3" in query.lower() and "bucket" in query.lower():
                print("\nExecuting S3 bucket listing...")
                aws_result = list_s3_buckets.invoke({})
                print("Result:", aws_result)
            elif "route 53" in query.lower() or "hosted zone" in query.lower():
                print("\nExecuting Route 53 hosted zones listing...")
                aws_result = list_route53_hosted_zones.invoke({})
                print("Result:", aws_result)
            elif "ec2" in query.lower() and "10.0.1.112" in query:
                print("\nGetting EC2 instance size...")
                aws_result = get_ec2_instance_size.invoke({"instance_ip": "10.0.1.112"})
                print("Result:", aws_result)
            elif "iam" in query.lower() and "take-home-coding" in query.lower():
                print("\nGetting IAM user permissions...")
                aws_result = get_user_permissions.invoke({"user_name": "take-home-coding"})
                print("Result:", aws_result)
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing Enhanced AWS Agent...")
    test_aws_agent()
