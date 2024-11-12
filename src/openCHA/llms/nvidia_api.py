from typing import Any
from typing import Dict
from typing import List

from openCHA.llms import BaseLLM
from openCHA.utils import get_from_dict_or_env
from pydantic import model_validator
import os


class NvidiaLLM(BaseLLM):
    """
    **Description:**

        An implementation of the Nvidia APIs. `Nvidia API <https://>`_
    """

    #  track max tokens for each model
    models: Dict = {
        "meta/llama3-8b-instruct": 8192,
    }

    api_key: str = ""
    llm_model: Any = None
    max_tokens: int = 150

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """

        Args:
            cls (type): The class itself.
            values (Dict): The dictionary containing the values for validation.
        Return:
            Dict: The validated dictionary with updated values.
        Raise:
            ValueError: If the anthropic python package cannot be imported.

        """

        values["api_key"] = os.getenv('NVIDIA_API_KEY', "no NVIDIA API key found!")

        try:
            from openai import OpenAI
            print(f"instantiated NvidiaLLM with key {values["api_key"]}")
            values["llm_model"] = OpenAI(base_url = "https://integrate.api.nvidia.com/v1", api_key = values["api_key"] )
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values
    


    def get_model_names(self) -> List[str]:
        """
            Get a list of available model names.

        Return:
            List[str]: A list of available model names.

        """

        return self.models.keys()
    


    def is_max_token(self, model_name, query) -> bool:
        """
            Check if the token count of the query exceeds the maximum token count for the specified model.

            It calculates the number of tokens from tokenizing the input query and compares it with the maximum allowed tokens for the model.
            If the number of tokens is greater than the maximum, it returns True.

        Args:
            model_name (str): The name of the model.
            query (str): The query to check.
        Return:
            bool: True if the token count exceeds the maximum, False otherwise.

        """

        model_max_token = self.models[model_name]
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please install it with `pip install tiktoken`."
            )
        
        #  need to get encoders for Nvidia to pass to tiktoken.get_encoding()
        encoder = "gpt2"
        if model_name in (
            "text-davinci-003",
            "text-davinci-002",
        ):
            encoder = "p50k_base"
        if model_name.startswith("code"):
            encoder = "p50k_base"

        enc = tiktoken.get_encoding(encoder)
        tokenized_text = enc.encode(query)


        return model_max_token < len(tokenized_text)
    



    def _parse_response(self, response) -> str:
        """
            Parse the response object and return the generated completion text.

        Args:
            response (object): The response object.
        Return:
            str: The generated completion text.


        """

        return response.choices[0].message.content
    


    def _prepare_prompt(self, prompt) -> Any:
        """
            Prepare the prompt by combining the human and AI prompts with the input prompt.
        Args:
            prompt (str): The input prompt.
        Return:
            Any: The prepared prompt.
        """

        return [{"role": "system", "content": prompt}]
    


    def generate(self, query: str, **kwargs: Any) -> str:
        """
            Generate a response based on the provided query.
        Args:
            query (str): The query to generate a response for.
            **kwargs (Any): Additional keyword arguments.
        Return:
            str: The generated response.
        Raise:
            ValueError: If the model name is not specified or is not supported.
        """

        model_name = "meta/llama3-8b-instruct"
        if "model_name" in kwargs:
            model_name = kwargs["model_name"]
        if model_name not in self.get_model_names():
            raise ValueError(
                "model_name is not specified or Nvidia does not support provided model_name"
            )
        stop = kwargs["stop"] if "stop" in kwargs else None
        max_tokens = (
            kwargs["max_tokens"]
            if "max_tokens" in kwargs
            else self.max_tokens
        )
        print("here", max_tokens, model_name)

        self.llm_model.api_key = self.api_key
        query = self._prepare_prompt(query)
        response = self.llm_model.chat.completions.create(
            model=model_name,
            messages=query,
            max_tokens=max_tokens,
            stop=stop,
        )
        return self._parse_response(response)
