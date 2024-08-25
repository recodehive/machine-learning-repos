import express from 'express';
import fetch from 'node-fetch';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const GITHUB_TOKEN = process.env.GITHUB_TOKEN
const app = express();
const port = 3000;

app.use(express.static(path.join(__dirname, '../')));
app.get('/api/github/repos/subdir', async (req, res) => {
    const dirName = req.query.dir;
    if (!dirName) {
        return res.status(400).json({ error: "Directory name is required" });
    }

    try {
        const response = await fetch(`https://api.github.com/repos/recodehive/machine-learning-repos/contents/${dirName}`, {
            headers: {
                Authorization: `Bearer ${GITHUB_TOKEN}`, 
            },
        });
        if (!response.ok) {
            const errorDetails = await response.text();
            throw new Error(`GitHub API error: ${response.status} - ${response.statusText}: ${errorDetails}`);
        }

        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error(`Error fetching GitHub subdirectory contents for ${dirName}:`, error);
        res.status(500).json({ error: error.message });
    }
});
app.get('/api/github/repos', async (req, res) => {
    try {
        const response = await fetch('https://api.github.com/repos/recodehive/machine-learning-repos/contents/', {
            headers: {
                Authorization: `Bearer ${GITHUB_TOKEN}`, 
            },
        });
        if (!response.ok) {
            const errorDetails = await response.text();
            throw new Error(`GitHub API error: ${response.status} - ${response.statusText}: ${errorDetails}`);
        }

        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error('Error fetching GitHub directories:', error);
        res.status(500).json({ error: error.message });
    }
});
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../index.html'));
});
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
