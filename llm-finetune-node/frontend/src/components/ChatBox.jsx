import { useState } from "react";
import axios from "axios";

const ChatBox = () => {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/generate", { prompt });
      setResponse(res.data.response);
    } catch (error) {
      setResponse("Error fetching response", error);
    }
    setLoading(false);
  };

  return (
    <div className="chat-container">
      <h1>Fine-Tuned LLM Chat</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ask something..."
        />
        <button type="submit" disabled={loading}>
          {loading ? "Generating..." : "Ask"}
        </button>
      </form>
      {response && <p className="response">{response}</p>}
    </div>
  );
};

export default ChatBox;
