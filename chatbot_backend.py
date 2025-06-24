#1 import the ConversationSummaryBufferMemory, ConversationChain, ChatBedrock or ChatBedrockConverse Langchain Modules
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_aws import ChatBedrockConverse

#2a Write a function for invoking model- client connection with Bedrock with profile, model_id & Inference params- model_kwargs
def chat_model():
    model = ChatBedrockConverse(
        credentials_profile_name='default',
        # cross-region models might require to adapt the SCPs: https://aws.amazon.com/blogs/machine-learning/enable-amazon-bedrock-cross-region-inference-in-multi-account-environments/
        # here I'm using a single region alternative
        model="amazon.titan-text-lite-v1", 
        provider='amazon',
        temperature=0.1,
        region_name="eu-west-1", 
        max_tokens=1000
    )
    return model
#3 Create a Function for ConversationBufferMemory (llm and max token limit)
def memory():
    conversation_memory = ConversationSummaryBufferMemory(
        llm=chat_model(),
        max_token_limit=1000
    )
    return conversation_memory
#4 Create a Function for Conversation Chain - Input text + Memory
# NEW: Define a clearer, more direct prompt template
from langchain_core.prompts import PromptTemplate
_template = """
Answer to the latest Human input as the AI, without simulating further rounds of conversation.

Current conversation:
{history}
Human: {input}
AI:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=_template)

# Using smarter models might result in better responses
def converse(input_text, memory):
    conversation_chain=ConversationChain(
        llm=chat_model(),
        memory=memory,
        prompt=PROMPT,
        verbose=True
    )
    #5 Chat response using invoke (Prompt template)
    chat_reply = conversation_chain.invoke(input_text)
    return chat_reply['response']
#Links :
#https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html
#https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html