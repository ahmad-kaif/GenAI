import express from "express";
import cors from "cors";
import { pipeline } from "@xenova/transformers";

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

let generator;

// Load the fine-tuned model
async function loadModel() {
    console.log("Loading fine-tuned model...");
    generator = await pipeline("text-generation", "../models/fine_tuned_model");
    console.log("Model loaded successfully!");
}
loadModel();

// API to generate responses
app.post("/generate", async (req, res) => {
    const { prompt } = req.body;

    if (!generator) {
        return res.status(500).json({ error: "Model is still loading, please wait..." });
    }

    try {
        const response = await generator(prompt, { max_length: 50 });
        res.json({ response: response[0].generated_text });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
