# Discovery package

Often LLMs are not well documented, this package is used to record workings to discover model properties.

For example, there is nothing in the Phi-4-mini-instruct chat_template which explains what tokens the model will output when it creates a tool call. It seems the only way to find this out, is to run inference and analyse the result.