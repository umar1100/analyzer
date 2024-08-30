import openai
#import pyodbc
import os
#from dotenv import find_dotenv, load_dotenv
import time
import logging
import pandas as pd
from datetime import datetime
import requests
import json
import streamlit as st

print('start')
#load_dotenv()

news_api_key = "2b4fff9ec9854806a3d24574b2d64b39"

client = openai.OpenAI(api_key="sk-oIovQpqLTCEPwyExB5rbeDHew_H2xpO1ppS64vTHuyT3BlbkFJhYL7jafNcV6CwW66JyurX8mQ-4JaS2WxHivByhtT0A")
model = "gpt-4o-mini"


def get_news(topic):
    url = (
        f"https://newsapi.org/v2/everything?q={topic}&apiKey={news_api_key}&pageSize=5"
    )

    try:
        response = requests.get(url)
        if response.status_code == 200:
            news = json.dumps(response.json(), indent=4)
            news_json = json.loads(news)

            data = news_json

            # Access all the fields == loop through
            status = data["status"]
            total_results = data["totalResults"]
            articles = data["articles"]
            final_news = []

            # Loop through articles
            for article in articles:
                source_name = article["source"]["name"]
                author = article["author"]
                title = article["title"]
                description = article["description"]
                url = article["url"]
                content = article["content"]
                title_description = f"""
                   Title: {title}, 
                   Author: {author},
                   Source: {source_name},
                   Description: {description},
                   URL: {url}
            
                """
                final_news.append(title_description)

            return final_news
        else:
            return []

    except requests.exceptions.RequestException as e:
        print("Error occured during API Request", e)

# Function to get weather information
def get_weather(city):
    # Replace with your actual weather API key
    weather_api_key = "5e407da2ae138dd42074bf8d12617aad"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            weather = json.dumps(response.json(), indent=4)
            weather_json = json.loads(weather)

            # Extract relevant data
            main = weather_json["main"]
            weather_description = weather_json["weather"][0]["description"]
            temperature = main["temp"]
            pressure = main["pressure"]
            humidity = main["humidity"]
            wind_speed = weather_json["wind"]["speed"]
            city_name = weather_json["name"]

            # Format the final weather information
            final_weather = f"""
                Weather in {city_name}:
                Description: {weather_description}
                Temperature: {temperature}°C
                Pressure: {pressure} hPa
                Humidity: {humidity}%
                Wind Speed: {wind_speed} m/s
            """

            return final_weather
        else:
            return f"Error: Unable to fetch weather data for {city}."

    except requests.exceptions.RequestException as e:
        print("Error occurred during API Request", e)
        return "Error: Unable to fetch weather data due to a request error."




class AssistantManager:
    _instance = None  # Class-level attribute to hold the single instance
    thread_id = None
    assistant_id = "asst_q4C9apv9tnQuM7XwCDC3u3Ba"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AssistantManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, model: str = model):
        if not hasattr(self, "initialized"):  # Check if the instance is already initialized
            self.client = client
            self.model = model
            self.assistant = None
            self.thread = None
            self.run = None
            self.summary = None
            if not AssistantManager.thread_id:
                self.create_thread() # Creating a thread each time a program is run
                print(f"ThreadID initialized::: {self.thread.id}")
            

            # Retrieve existing assistant and thread if IDs are already set
            if AssistantManager.assistant_id:
                self.assistant = self.client.beta.assistants.retrieve(
                    assistant_id=AssistantManager.assistant_id
                )
            if AssistantManager.thread_id:
                self.thread = self.client.beta.threads.retrieve(
                    thread_id=AssistantManager.thread_id
                )
        return

    def create_assistant(self, name, instructions, tools, tool_resources):
        if not self.assistant:
            assistant_obj = self.client.beta.assistants.create(
                name=name, instructions=instructions, tools=tools, model=self.model, tool_resources= tool_resources
            )
            AssistantManager.assistant_id = assistant_obj.id
            self.assistant = assistant_obj
            print(f"AssisID:::: {self.assistant.id}")

    def create_thread(self):
        if not AssistantManager.thread_id:
            thread_obj = self.client.beta.threads.create()
            AssistantManager.thread_id = thread_obj.id
            self.thread = thread_obj

    def add_message_to_thread(self, role, content):
        if self.thread:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id, role=role, content=content
            )

    def run_assistant(self, instructions):
        if self.thread and self.assistant:
            self.run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                instructions=instructions,
            )

    def upload_file(self):
        if self.thread and self.assistant:
            file = client.files.create(
            file=open("data1.txt", "rb"),
            purpose='assistants'
            )
            print("The file id is : ", file.id)
            openai.beta.assistants.files.create(self.assistant.id, {
            "file_ids": [file.id],
        })

    def process_message(self):
        if self.thread:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            summary = []

            last_message = messages.data[0]
            role = last_message.role
            response = last_message.content[0].text.value
            summary.append(response)

            self.summary = "\n".join(summary)
            print(f"SUMMARY-----> {role.capitalize()}: ==> {response}")



    def call_required_functions(self, required_actions):
        if not self.run:
            return
        
        tool_outputs = []
        tool_call_ids = [action["id"] for action in required_actions.get("tool_calls", [])]

        for action in required_actions.get("tool_calls", []):
            func_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if func_name == "get_news":
                output = get_news(topic=arguments.get("topic", ""))
                final_str = "".join(output) if isinstance(output, list) else output
                tool_outputs.append({"tool_call_id": action["id"], "output": final_str})
            elif func_name == "get_weather":
                output = get_weather(city=arguments.get("city", ""))
                tool_outputs.append({"tool_call_id": action["id"], "output": output})


        # Ensure all tool calls are addressed
        for tool_call_id in tool_call_ids:
            if not any(tool_call["tool_call_id"] == tool_call_id for tool_call in tool_outputs):
                tool_outputs.append({"tool_call_id": tool_call_id, "output": ""})

        if tool_outputs:
            print("Submitting outputs back to the Assistant...")
            try:
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id, run_id=self.run.id, tool_outputs=tool_outputs
                )
            except openai.OpenAIError as e:
                print(f"Error submitting tool outputs: {e}")


    # for streamlit
    def get_summary(self):
        return self.summary

    def wait_for_completion(self):
        if self.thread and self.run:
            while True:
                time.sleep(5)
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id, run_id=self.run.id
                )
                print(f"RUN STATUS:: {run_status.model_dump_json(indent=4)}")

                if run_status.status == "completed":
                    # Process the assistant's response if completed
                    self.process_message()
                    break
                elif run_status.status == "requires_action":
                    required_actions = run_status.required_action.submit_tool_outputs.model_dump()
                    
                    if "tool_calls" in required_actions and required_actions["tool_calls"]:
                        print("FUNCTION CALLING NOW...")
                        self.call_required_functions(required_actions)
                    else:
                        # If no function calls are needed, process the response directly
                        self.process_message()
                        break



    # Run the steps
    def run_steps(self):
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id, run_id=self.run.id
        )
        print(f"Run-Steps::: {run_steps}")
        return run_steps.data

# New Function to Connect to SQL and Fetch Data
def fetch_sql_data():
    #Server Parameters
    server = 'ips-bi.database.windows.net'
    database = 'SMG_BI'
    username = 'IPS_Admin'
    password = 'KdnLx@Xz5LjBT6T'

    #Connecting to Database
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';ENCRYPT=yes;UID='+username+';PWD='+ password)

    #Reading the database
    df = pd.read_sql("SELECT o.ordnum,o.clientid,o.invdate,o.ordertype,o.category,o.priority,o.dne,o.status,o.invtotal,o.tax,o.taxrate,o.billcompany,o.billcity,o.billstate,p.VendorCost,p.Profit,p.Perc FROM [dbo].[BI_Orders_DT_View] As o  JOIN [dbo].[BI_OrderProfit_DT] AS p ON o.Ordnum = p.Ordnum", cnxn)
    print("The number of data rows are: ", len(df))
    df.to_csv('data1.txt', sep=',', index=False)

    return 

def initialize_manager():
    """Initialize the AssistantManager and create an assistant if not already done."""
    if 'manager' not in st.session_state:
        st.session_state.manager = AssistantManager()

        if st.session_state.manager.assistant is None:
            st.session_state.manager.create_assistant(
                name="News, Weather and Data Analyzer",
                instructions="""
                You are a personal assistant capable of summarizing news articles, providing weather information for any city, and analyzing data from provided files.
                When using the code interpreter, load the data from the provided files and process it according to the user's query.
                """,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_news",
                            "description": "Get the list of articles/news for the given topic",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "topic": {
                                        "type": "string",
                                        "description": "The topic for the news, e.g. bitcoin",
                                    }
                                },
                                "required": ["topic"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the current weather information for the given city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "The name of the city for which to retrieve the weather, e.g., Calgary",
                                    }
                                },
                                "required": ["city"],
                            },
                        },
                    },
                    {
                        "type": "code_interpreter"
                    }
                ],
                tool_resources={"code_interpreter": {
                    "file_ids": ["file-fQhHTukUFHJfpc1KYsxToBXc"]
                }}
            )

def process_request(instructions):
    """Helper function to process the request with the assistant."""
    manager = st.session_state.manager
    manager.add_message_to_thread(role="user", content=instructions)
    manager.run_assistant(instructions="Provide answers to users query")
    manager.wait_for_completion()
    return manager.get_summary()

def main():
    initialize_manager()

    # Initialize button state and processing flag if not already set
    if 'buttons_disabled' not in st.session_state:
        st.session_state.buttons_disabled = False
        st.session_state.processing = False
        st.session_state.summary = None
        st.session_state.last_action = None  # Track the last action

    # Title
    st.title("Data Analyzer")

    # Form for user input
    with st.form(key="user_input_form"):
        instructions = st.text_input("Enter Instructions:")

        # Button Columns for better layout
        col1, col2, col3 = st.columns(3)
        submit_button = col1.form_submit_button(
            label="Run User Query", 
            disabled=st.session_state.buttons_disabled
        )
        analysis_button = col2.form_submit_button(
            label="Data Summary", 
            disabled=st.session_state.buttons_disabled
        )
        optimizer_button = col3.form_submit_button(
            label="Optimizer", 
            disabled=st.session_state.buttons_disabled
        )

        if submit_button or analysis_button or optimizer_button:
            if not st.session_state.processing:
                # Disable all buttons while processing
                st.session_state.buttons_disabled = True
                st.session_state.processing = True

                # Determine which button was pressed
                if submit_button:
                    st.session_state.last_action = "submit"
                    st.session_state.summary = process_request(f"{instructions}?")
                    
                
                elif analysis_button:
                    st.session_state.last_action = "analysis"
                    st.session_state.summary = process_request("Summarize the given data available to you")

                elif optimizer_button:
                    st.session_state.last_action = "optimizer"
                    st.session_state.summary = process_request(
                        "Look at vendors that receive a large amount of revenue from this company, give me some options for negotiating lower rates with them"
                    )
                
                # Re-enable buttons and reset processing flag after processing
                st.session_state.buttons_disabled = False
                st.session_state.processing = False

        # Display the summary if available
        if st.session_state.summary:
            st.write(st.session_state.summary)
            # Reset the summary after displaying it
            st.session_state.summary = None

if __name__ == "__main__":
    main()
