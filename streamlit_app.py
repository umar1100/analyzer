import openai
import pyodbc
import os
#from dotenv import find_dotenv, load_dotenv
import time
import logging
import pandas as pd
from datetime import datetime
import requests
import json
import streamlit as st
import requests
import requests
from datetime import datetime, timedelta


#load_dotenv()

news_api_key = st.secrets["news_api_key"]

client = openai.OpenAI(api_key=st.secrets["open_api_key"],)
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
    weather_api_key = st.secrets["weather_api_key"]
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




def google_search(query):
    url = "https://google-api31.p.rapidapi.com/websearch"
    headers = {
        "x-rapidapi-key": st.secrets["x-rapidapi-key"],
        "x-rapidapi-host": "google-api31.p.rapidapi.com",
        "Content-Type": "application/json"
    }
    payload = {
        "text": query,
        "safesearch": "off",
        "timelimit": "",
        "region": "wt-wt",
        "max_results": 5
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        results = response.json()

        # Check if the 'result' key exists
        if 'result' in results:
            output = []
            for result in results['result'][:5]:
                title = result.get('title')
                link = result.get('href')
                snippet = result.get('body')
                output.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n")
            return '\n'.join(output)
        else:
            return "No results found."
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error: Unable to fetch search results."



class AssistantManager:
    _instance = None  # Class-level attribute to hold the single instance
    thread_id = None
    assistant_id = st.secrets["assistant_id"]


    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AssistantManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, model: str = model):
        if not hasattr(self, "initialized"):
            self.client = client
            self.model = model
            self.assistant = None
            self.thread = None
            self.run = None
            self.summary = None

            if not AssistantManager.thread_id:
                self.create_thread()

            if AssistantManager.assistant_id:
                self.assistant = self.client.beta.assistants.retrieve(
                    assistant_id=AssistantManager.assistant_id
                )

            self.initialized = True

    def create_assistant(self, name, instructions, tools, tool_resources):
        if not self.assistant:
            assistant_obj = self.client.beta.assistants.create(
                name=name, instructions=instructions, tools=tools, model=self.model, tool_resources=tool_resources
            )
            AssistantManager.assistant_id = assistant_obj.id
            self.assistant = assistant_obj
            print(f"Assistant ID: {self.assistant.id}")

    def create_thread(self):
        if not AssistantManager.thread_id:
            thread_obj = self.client.beta.threads.create()
            AssistantManager.thread_id = thread_obj.id
            self.thread = thread_obj
            print(f"Created new thread with ID: {self.thread.id}")




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



            


    def process_message(self):
        if self.thread:
            if self.file:  # Ensure self.file contains a valid file ID
                print(f"Using file with ID: {self.file}, during the message processing")  # Debugging
                self.client.beta.threads.update(
                    thread_id=self.thread.id,
                    tool_resources={"code_interpreter": {"file_ids": [self.file]}}  # Use the file ID stored
                )
            else:
                print("No file available to attach to the thread.")

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
            elif func_name == "google_search":
                output = google_search(query=arguments.get("query", ""))
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


def fetch_sql_data():
    """Fetch data from SQL and create a new file."""
    # Server Parameters
    server = st.secrets["server"]
    database = st.secrets["database"]
    username = st.secrets["username"]
    password = st.secrets["password"]

    # Connecting to Database
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';ENCRYPT=yes;UID='+username+';PWD='+password)

    # Reading the database
    df = pd.read_sql("SELECT o.ordnum,o.clientid,o.invdate,o.ordertype,o.category,o.priority,o.dne,o.status,o.invtotal,o.tax,o.taxrate,o.billcompany,o.billcity,o.billstate,p.VendorCost,p.Profit,p.Perc FROM [dbo].[BI_Orders_DT_View] As o  JOIN [dbo].[BI_OrderProfit_DT] AS p ON o.Ordnum = p.Ordnum", cnxn)
    print("The number of data rows are: ", len(df))
    df.to_csv('data1.txt', sep=',', index=False)


import requests
from datetime import datetime, timedelta, timezone

def manage_files(api_key, new_file_path):
    """Delete old files and upload the new file."""
    # API endpoints
    list_files_url = "https://api.openai.com/v1/files"
    api_key = st.secrets["open_api_key"]
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Step 1: Get the list of files
    response = requests.get(list_files_url, headers=headers)
    if response.status_code == 200:
        file_list = response.json().get("data", [])
        print(f"Found {len(file_list)} files.")

        # Current time in UTC
        now = datetime.now(timezone.utc)

        # Step 2: Filter files with purpose "assistants" and delete those older than 2 hours
        for file_info in file_list:
            file_id = file_info.get("id")
            file_purpose = file_info.get("purpose")
            file_created_at = file_info.get("created_at")

            if file_id and file_purpose == "assistants":
                # Convert the created_at timestamp to a timezone-aware datetime object
                file_created_time = datetime.fromtimestamp(file_created_at, tz=timezone.utc)
                print('file creation time: ', file_created_time)
                # Check if the file is older than 10 hours
                if now - file_created_time > timedelta(hours=10):
                    delete_url = f"{list_files_url}/{file_id}"
                    delete_response = requests.delete(delete_url, headers=headers)
                    print('Delete URL', delete_url)
                    if delete_response.status_code == 204:
                        print(f"Deleted file with ID: {file_id}")
                    else:
                        print(f"Failed to delete file with ID: {file_id}, Status Code: {delete_response.status_code}")

        # Step 3: Upload a new file
        upload_url = "https://api.openai.com/v1/files"
        with open(new_file_path, 'rb') as file_to_upload:
            files = {
                "file": (new_file_path, file_to_upload, "text/plain"),
            }
            print(files)
            data = {
                "purpose": "assistants"  # Set the purpose to "assistants"
            }
            upload_response = requests.post(upload_url, headers=headers, files=files, data=data)
            
            if upload_response.status_code == 200:
                uploaded_file = upload_response.json()
                print(f"Uploaded new file with ID: {uploaded_file['id']}")
                return uploaded_file['id']
            else:
                print(f"Failed to upload file, Status Code: {upload_response.status_code} - {upload_response.text}")
                return None
    else:
        print(f"Failed to retrieve file list, Status Code: {response.status_code}")
        return None


def initialize_manager():
    """Initialize the AssistantManager and create an assistant if not already done."""
    if 'manager' not in st.session_state:
        st.session_state.manager = AssistantManager()

        # Fetch latest data and create a new file
        fetch_sql_data()

        # Check if the file has already been uploaded
        if 'uploaded_file_id' not in st.session_state:
            # Manage files: delete old files and upload the new one
            api_key = st.secrets["open_api_key"]
            new_file_path = "data1.txt"
            file_id = manage_files(api_key, new_file_path)

            if file_id:
                # Store the uploaded file's ID in session state so we don't re-upload it
                st.session_state.uploaded_file_id = file_id

                # Create or update the assistant with the new file
                if st.session_state.manager.assistant is None:
                    st.session_state.manager.create_assistant(
                        name="News, Weather, real time information and Data Analyzer",
                        instructions="""
                        You are a personal assistant capable of summarizing news articles, providing weather information for any city, getting real-time information, and analyzing data from provided files.
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
                                "type": "function",
                                "function": {
                                    "name": "google_search",
                                    "description": "Get search results for the given query including current information required",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "query": {
                                                "type": "string",
                                                "description": "The search query, e.g., 'Python programming language'"
                                            }
                                        },
                                        "required": ["query"]
                                    }
                                }
                            },
                            {
                                "type": "code_interpreter"
                            }
                        ]
                    )
                    print('Created new assistant')
                else:
                    print("Assistant already created. Skipping re-upload.")
                    print(f"Using uploaded file with ID: {st.session_state.uploaded_file_id}")
                    st.session_state.manager.file = file_id  # Assign file ID if the assistant already exists
        else:
            # Use the already uploaded file ID
            st.session_state.manager.file = st.session_state.uploaded_file_id
            print(f"Using previously uploaded file with ID: {st.session_state.uploaded_file_id}")


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

        # Layout with 2 rows of 3 buttons each
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)

        # First row with 3 buttons
        with row1_col1:
            optimizer_button = st.form_submit_button(
                label="Optimizer", 
                disabled=st.session_state.buttons_disabled
            )
        with row1_col2:
            analysis_button = st.form_submit_button(
                label="Data Summary", 
                disabled=st.session_state.buttons_disabled
            )
        with row1_col3:
            submit_button = st.form_submit_button(
                label="Run User Query", 
                disabled=st.session_state.buttons_disabled
            )

        # Second row with 3 buttons
        with row2_col1:
            Work_Forecast = st.form_submit_button(
                label="Work Forecast", 
                disabled=st.session_state.buttons_disabled
            )
        with row2_col2:
            Negotiator = st.form_submit_button(
                label="Negotiator", 
                disabled=st.session_state.buttons_disabled
            )
        with row2_col3:
            Opportunities = st.form_submit_button(
                label="Opportunities", 
                disabled=st.session_state.buttons_disabled
            )

        if any([submit_button, analysis_button, optimizer_button, Work_Forecast, Negotiator, Opportunities]):
            if not st.session_state.processing:
                st.session_state.processing = True
                # Determine which button was clicked
                if submit_button:
                    st.session_state.last_action = "Run User Query"
                elif analysis_button:
                    st.session_state.last_action = "Data Summary"
                elif optimizer_button:
                    st.session_state.last_action = "Optimizer"
                elif Work_Forecast:
                    st.session_state.last_action = "Work Forecast"
                elif Negotiator:
                    st.session_state.last_action = "Negotiator"
                elif Opportunities:
                    st.session_state.last_action = "Opportunities"

                # Handle button clicks
                if st.session_state.last_action == "Run User Query" and instructions:
                    st.session_state.summary = process_request(instructions)
                elif st.session_state.last_action == "Data Summary":
                    st.session_state.summary = process_request("Provide a summary of the data")
                elif st.session_state.last_action == "Optimizer":
                    st.session_state.summary = process_request("Optimize the data")
                elif st.session_state.last_action == "Work Forecast":
                    st.session_state.summary = process_request("Forecast the work based on data")
                elif st.session_state.last_action == "Negotiator":
                    st.session_state.summary = process_request("Provide negotiation strategies based on data")
                elif st.session_state.last_action == "Opportunities":
                    st.session_state.summary = process_request("Identify opportunities in the data")
                
                st.session_state.processing = False

    # Display the results
    if st.session_state.summary:
        st.subheader("Summary:")
        st.write(st.session_state.summary)

if __name__ == "__main__":
    main()