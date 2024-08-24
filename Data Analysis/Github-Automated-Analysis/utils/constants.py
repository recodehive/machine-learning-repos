from pathlib import Path

APP_NAME = "Github Automated Analysis Tool"
MODEL = "gpt-3.5-turbo"
PAGE_ICON = "🤖"

K = 10
FETCH_K = 20
CHUNK_SIZE = 1000
TEMPERATURE = 0.5
MAX_TOKENS = 3000
ENABLE_ADVANCED_OPTIONS = True

DATA_PATH = Path.cwd() / "data"




OPENAI_HELP = """
You can sign-up for OpenAI's API [here](https://openai.com/blog/openai-api).\n
Once you are logged in, you find the API keys [here](https://platform.openai.com/account/api-keys)
"""

ACTIVELOOP_HELP = """
You can create an Activeloop account (including 200GB of free database storage) [here](https://www.activeloop.ai/).\n
Once you are logged in, you find the API token [here](https://app.activeloop.ai/profile/gustavz/apitoken).\n
The organisation name is your username, or you can create new organisations [here](https://app.activeloop.ai/organization/new/create)
"""