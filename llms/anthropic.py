from __future__ import annotations
from utils import get_from_dict_or_env
from llms.llm import BaseLLM
from typing import Any, List, Dict
from pydantic import model_validator


class AntropicLLM(BaseLLM):
    models: Dict = {
        "claude-2": 100000,
    }
    llm_model: Any = None
    api_key: str = ""
    HUMAN_PROMPT: str = ""
    AI_PROMPT: str = ""

    @model_validator(mode='before')
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validate that api key and python package exists in environment.

        This method validates the environment by checking the existence of the API key and required Python packages.
        It retrieves the API key from either the "anthropic_api_key" key in the "values" dictionary or from the "ANTHROPIC_API_KEY" environment variable.
        It also imports the required packages and assigns the appropriate values to the class attributes.

        Args:
            cls (object): The class itself.
            values (Dict): The dictionary containing the values for validation.
        Return:
            Dict: The validated dictionary with updated values.
        Raise:
            ValueError: If the anthropic python package cannot be imported.



        Example:
            .. code-block:: python

                from langchain import ReActChain, OpenAI
                react = ReAct(llm=OpenAI())

        """

        anthropic_api_key = get_from_dict_or_env(
            values, "anthropic_api_key", "ANTHROPIC_API_KEY"
        )
        values["api_key"] = anthropic_api_key
        try:
            from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT

            values["llm_model"] = AsyncAnthropic
            values["HUMAN_PROMPT"] = HUMAN_PROMPT
            values["AI_PROMPT"] = AI_PROMPT
        except ImportError:
            raise ValueError(
                "Could not import anthropic python package. "
                "Please install it with `pip install anthropic`."
            )
        return values

    def get_model_names(self) -> List[str]:
        """
        Get a list of available model names.

        Return:
            List[str]: A list of available model names.



        Example:
            .. code-block:: python

                from langchain import ReActChain, OpenAI
                react = ReAct(llm=OpenAI())

        """

        return self.models.keys()

    def is_max_token(self, model_name, query) -> bool:
        """
        Check if the token count of the query exceeds the maximum token count for the specified model.

        Args:
            model_name (str): The name of the model.
            query (str): The query to check.
        Return:
            bool: True if the token count exceeds the maximum, False otherwise.



        Example:
            .. code-block:: python

                from langchain import ReActChain, OpenAI
                react = ReAct(llm=OpenAI())

        """

        model_max_token = self.models[model_name]
        token_count = self.llm_model(api_key=self.api_key).count_tokens(query)
        return model_max_token < token_count

    def parse_response(self, response) -> str:
        """
        Parse the response object and return the generated completion text.

        Args:
            response (object): The response object.
        Return:
            str: The generated completion text.



        Example:
            .. code-block:: python

                from langchain import ReActChain, OpenAI
                react = ReAct(llm=OpenAI())

        """

        return response.completion

    def prepare_prompt(self, prompt) -> Any:
        """
        Prepare the prompt by combining the human and AI prompts with the input prompt.

        Args:
            prompt (str): The input prompt.
        Return:
            Any: The prepared prompt.



        Example:
            .. code-block:: python

                from langchain import ReActChain, OpenAI
                react = ReAct(llm=OpenAI())

        """

        return f"{self.HUMAN_PROMPT} {prompt}{self.AI_PROMPT}"

    def generate(
            self,
            query: str,
            **kwargs: Any
    ) -> str:
        """
        Generate a response based on the provided query.

        Args:
            query (str): The query to generate a response for.
            **kwargs (Any): Additional keyword arguments.
        Return:
            str: The generated response.
        Raise:
            ValueError: If the model name is not specified or is not supported.



        Example:
            .. code-block:: python

                from langchain import ReActChain, OpenAI
                react = ReAct(llm=OpenAI())

        """

        model_name = "claude-2"
        if "model_name" in kwargs:
            model_name = kwargs["model_name"]
        if model_name not in self.get_model_names():
            raise ValueError(
                "model_name is not specified or OpenAI does not support provided model_name"
            )

        max_token = kwargs["max_token"] if "max_token" in kwargs else 32000
        query = self.prepare_prompt(query)
        response = self.llm_model(api_key=self.api_key).completions.create(
            model=model_name,
            max_tokens_to_sample=max_token,
            prompt=query,
        )
        return self.parse_response(response)
