import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const router = express.Router();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

router.get('/', (req, res) => {     // root path
    res.sendFile(path.resolve(__dirname, '../index.html'));
});

router.get('/about', (req, res) => {
    res.sendFile(path.resolve(__dirname, '../about.html'));
});

router.get('/organization', (req, res) => {
    res.sendFile(path.resolve(__dirname, '../organization.html'));
});

router.get('/faq', (req, res) => {
    res.sendFile(path.resolve(__dirname, '../faq.html'));
});

router.get('/contact', (req, res) => {
    res.sendFile(path.resolve(__dirname, '../contact.html'));
});

router.get('*',(req, res) => {      // routes to handle invalid url path
    res.status(404).sendFile(path.resolve(__dirname, '../404.html'));
})


export default router;