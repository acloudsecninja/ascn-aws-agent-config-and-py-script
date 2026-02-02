# ascn-langchain-acsn-acct-configuration

## Python pull-info-from-aws-acct.py Steps ## 

Purpose: This code will pull directly from AWS APIs and pull down the requested information using python scripts.

## Prerequisites

Before you start, ensure you have the following:

    AWS Account
    AWS SDK for Python (Boto3)
    Langchain library
    OpenAI LLM Model Configuration / API Key
    Windows 11
    Python 3.13.9


Step 1: On Windows Open Git Bash terminal within Visual Studio Code

Step 2: Install Required Libraries

```bash
pip install -r requirements.txt
```

Step 3: Setup AWS Credentials

```bash

##Example Using simple AWS ACCESS and SECRET KEY
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
region = us-east-2
###
```

## Note: These Credentials will be found in ~/.aws/credentials or .env depneding on your perferred method.

Step 4: Create a local .env file on Windows

- Create a local .env file in github so you can set your OPENAIKEY

OPENAI_API_KEY=your_openai_api_key_here


## Critical_NOTE: These Credentials will be found in .env file ,but not uploaded to Github since .gitignore is set

Step 5: Run the command pull-info-from-aws-acct.py Python Script to setup boto3 (AWS).

 - Run the command to start and manage the bot that will interact with AWS APIs.

 ```bash
 ./pull-info-from-aws-acct.py
 ```

OR

Run the command which will use an AWS Agent to interact with AWS API services and return results as expected.

```bash
./ascn-aws-agent.py
```

- Created and Maintained by - A Cloud Security Ninja LLC - 

- For Information contact us at @ (https://www.acloudsec.ninja/booking-calendar/free-15-minute-consultation)