# RAG Chatbot using ReAct Agent

This project aims to develop a Retrieval-Augmented Generation (RAG) chatbot utilizing the ReAct Agent framework. The chatbot is designed to provide intelligent and context-aware responses by combining retrieval-based and generative approaches. The implementation leverages **LangChain** for building conversational agents and **Streamlit** for creating an interactive user interface.

## Features

- **Retrieval-Augmented Generation**: Leverages external knowledge sources to enhance response accuracy.
- **ReAct Agent Framework**: Implements reasoning and acting capabilities for dynamic interactions.
- **LangChain Integration**: Simplifies the development of conversational AI workflows.
- **Streamlit Interface**: Provides an intuitive and user-friendly web interface.
- **Context-Aware Responses**: Ensures meaningful and relevant conversations.

## Getting Started

Follow the steps below to set up and run the project:

1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd RAG-Chatbot-using-ReAct-Agent
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```bash
        streamlit run hello.py
        ```

    ## Project Structure

    - `/src`: Contains the source code for the chatbot.
    - `/data`: Includes datasets or external knowledge sources.
    - `/docs`: Documentation and resources for the project.
    - `hello.py`: Entry point for the Streamlit application utilizing the OpenAI model.

## Contributing

Contributions are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to the open-source community for providing tools and frameworks like LangChain and Streamlit that made this project possible.