import os
import streamlit as st
from openai import OpenAI
import weaviate
from rag.preprocess import extract_sections, fetch_text
from rag.utils import get_num_tokens, trim
import json



source_link = "https://scikit-learn.org/stable/modules/generated/"

client = weaviate.Client(
    url = "https://bhujith-weaviate-bjmg7g0q.weaviate.network",  # Replace with your endpoint
    auth_client_secret=weaviate.AuthApiKey(api_key=keys['WEAVIATE_API_KEY']),  # Replace w/ your Weaviate instance API key
    additional_headers = {
        "X-OpenAI-Api-Key": keys['OPENAI_API_KEY']  # Replace with your inference API key
    }
)

def make_clickable(link):
    return f'<a href="{link}" target="_blank">{link}</a>'

def get_top_answers_from_vector_db(input_query,num_chunks):
    n_top_results = 5
    response = (
        client.query
        .get("ScikitLearnDocumentation", ["text", "source"])
        .with_near_text({"concepts":[input_query]})
        .with_limit(num_chunks)
        .do()
    ) 
    top_responses = [ans['text'] for ans in response['data']['Get']['ScikitLearnDocumentation'][:n_top_results]]
    top_sources = [ans['source'] for ans in response['data']['Get']['ScikitLearnDocumentation'][:n_top_results]]
    return top_responses,top_sources

def generate_response(llm,system_content,assistant_content,user_content):
    client = OpenAI(
        api_key=keys["OPENAI_API_KEY"]
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_content 
            },
            {
                "role": "assistant", 
                "content": assistant_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        model=llm,
        stream=False
    )
    return chat_completion.choices[0].message.content

class QueryAgent:
    def __init__(self, llm, max_context_length=4096, system_content="", assistant_content=""):
        
        # Context length (restrict input length to 50% of total context length)
        max_context_length = int(0.5*max_context_length)
        
        # LLM
        self.llm = llm
        self.context_length = max_context_length - get_num_tokens(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(self, query, num_chunks=5):
        # Get sources and context
        context,sources = get_top_answers_from_vector_db(query,num_chunks)
            
        # Generate response
        user_content = f"query: {query}, context: {context}"
        answer = generate_response(
            llm=self.llm,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=trim(user_content, self.context_length))

        # Result
        result = {
            "question": query,
            "sources": sources,
            "answer": answer,
            "llm": self.llm,
        }
        return result
    

    
st.title('🦜🔗 Scikit-Learn Documentation Search')
# Create a text input box for the user
query = st.text_input('Input your prompt here')

# If the user hits enter
if query:
    system_content = "Answer the query using the context provided. Be succinct."
    llm = "gpt-4"
    agent = QueryAgent(
        llm=llm,
        system_content=system_content
        )
    result = agent(query=query)
    print("\n\n", json.dumps(result, indent=2))
    # ...and write it out to the screen
    st.subheader('Answer')
    st.write(result['answer'])
    
    st.subheader('Sources')
    for link in result['sources']:
        hyperlink = source_link + link.split("\\")[-1]
        text='check out this [link]({link})'.format(link=hyperlink)
        st.markdown(text,unsafe_allow_html=True)
    


