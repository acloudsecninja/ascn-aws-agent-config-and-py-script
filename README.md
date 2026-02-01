# ascn-langchain-acsn-acct-configuration

Purpose: This code will setup an agent that can interface with AWS services and process text using an OpenAI model and give information based on the AWS account in question.

## Prerequisites

Before you start, ensure you have the following:

    AWS Account
    AWS SDK for Python (Boto3)
    Langchain library
    OpenAI LLM Model Configuration / API Key


Step 1: On Windows Open Git Bash terminal within Visual Studio Code

Step 2: Create a virtual python env

```bash
python -m venv myenv
```

```bash
./myenv/Scripts/activate
```

Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

Step 4: Setup AWS Credentials

```bash

##Example Using simple AWS ACCESS and SECRET KEY
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
region = us-east-2
###
```

## Note: These Credentials will be found in ~/.aws/credentials

Step 5: Create a local .env file on Windows (Similar for linux/mac)

- Create a local .env file using notepad and then setup the value in your notepad and save it in your local user folder. 

OPENAI_API_KEY=your_openai_api_key_here


## Critical_NOTE: These Credentials will be found in .env file ,but not uploaded to Github

Step 6: Use create_aws_chatbot.py Python Script to setup boto3 (AWS) and Langchain and Import OPENAI Model with specfic commands at python script

 - Run the command to start and manage the bot.

 ```bash
 ./create_aws_chatbot.py
```

Step 7: Once the bot is created you can then run the following commands to see if the bot will respond with the information you want.
