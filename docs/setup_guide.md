# Guide for setting environment up for development

 - locate to the root directory
 - create a new python environment
   - `python -m venv venv; ./venv/Scripts/activate`
 - run `pip install -e .` in command lines
 - If you want to setup the environment for api keys instead of manually put it every single time, set up the environmental variables with your open ai api keys
   - in windows you could do `setx OPENAI_API_KEY sk-xxxxxxxxx`
   - in macos/linux you could do `export OPENAI_API_KEY=sk-xxxxxxxxx`
   - The same goes for open router `OPENROUTER_API_KEY`
 - to test if everything is working, do the following:
   - go to interactive command lines in python `python`
    ```
    from nerif import SimpleChatAgent
    agent = SimpleChatAgent()
    agent.chat("Hello world!")
    >> 'Hello! How can I assist you today?'
    ```