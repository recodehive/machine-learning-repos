import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const router = express.Router();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rootDirectory = path.resolve(__dirname, '../');

router.get('/', (req, res) => {
    res.sendFile(path.join(rootDirectory, 'index.html'));
});

router.get('/about', (req, res) => {
    res.sendFile(path.join(rootDirectory, 'about.html'));
});

router.get('/organization', (req, res) => {
    res.sendFile(path.join(rootDirectory, 'organization.html'));
});

router.get('/faq', (req, res) => {
    res.sendFile(path.join(rootDirectory, 'faq.html'));
});

router.get('/contact', (req, res) => {
    res.sendFile(path.join(rootDirectory, 'contact.html'));
});

// Handle all other routes with a 404 page
router.get('*', (req, res) => {
    res.status(404).sendFile(path.join(rootDirectory, '404.html'));
});

export default router;
