#1 import the ConversationSummaryBufferMemory, ConversationChain, ChatBedrock or ChatBedrockConverse Langchain Modules
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_aws import ChatBedrockConverse

#2a Write a function for invoking model- client connection with Bedrock with profile, model_id & Inference params- model_kwargs
def demo_chatbot(messages):
    demo_llm = ChatBedrockConverse(
        credentials_profile_name='default',
        # cross-region models might require to adapt the SCPs: https://aws.amazon.com/blogs/machine-learning/enable-amazon-bedrock-cross-region-inference-in-multi-account-environments/
        # here I'm using a single region alternative
        model="amazon.titan-text-lite-v1", 
        provider='amazon',
        temperature=0.1,
        region_name="eu-west-1", 
        max_tokens=1000
    )
    return demo_llm.invoke(messages)
#2b Test out the LLM with invoke method 
messages = [
    {
        "role": "user",
        "content": [{"text": 'What is an LLM'}]
    }
]
response = demo_chatbot(messages)
print(response)
#3 Create a Function for ConversationBufferMemory (llm and max token limit)
#4 Create a Function for Conversation Chain - Input text + Memory
#5 Chat response using invoke (Prompt template)
#Links :
#https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html
#https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
#https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.htm