import * as fs   from 'fs-extra';
import * as path from 'path';
import * as uuid from 'uuid';
import * as proc from 'child_process';

import * as express from 'express';
import * as multer  from 'multer';

const port = 8080;
console.log(`[server] port: ${port}.`);

const uploadDir = path.resolve(__dirname, '../../public/upload/');
const resultDir = path.resolve(__dirname, '../../public/result/');
const maePyPath = path.resolve(__dirname, '../../mae.py');
fs.mkdirSync(uploadDir, { recursive: true });
fs.mkdirSync(resultDir, { recursive: true });
// fs.emptyDirSync(uploadDir);

const app    = express();
const upload = multer({ dest: uploadDir });
app.use(express.static('public'));
app.use(express.json());

app.post("/api/upload", upload.single("image"), async (req, res) => {
    const id = uuid.v1();
    fs.renameSync(req.file!.path, path.resolve(uploadDir, id));
    console.log(`[server] image uploaded: upload/${id}.`)
    res.json({ id }).status(200).send();
})

app.post("/api/run", (req, res) => {
    const id         = String(req.body["id"]);
    const modelId    = String(req.body["modelId"]);
    const modelName  =
        modelId === "0" ? "mae_visualize_vit_large_ganloss.pth" :
        modelId === "1" ? "celeba.pth" :
        modelId === "2" ? "places.pth" :
        "mae_visualize_vit_large_ganloss.pth";
    const maskBase64 = String(req.body["maskBase64"]);
    
    console.log(`[server] task accepted.`)
    console.log(`[server]    id: ${id}`);
    console.log(`[server]    chkp: ${modelName}`);
    console.log(`[server]    mask: ${maskBase64}`);

    const imgPath = path.resolve(uploadDir, id);
    const outDir  = path.resolve(resultDir, id);
    fs.mkdirSync(outDir, { recursive: true });

    const process = proc.spawn(`py ${maePyPath} `
        + `-i ${imgPath}   -o ${outDir} `
        + `-c ${modelName} -m ${maskBase64}`, {
        shell: true,
    });
    process.stdout.on('data', (chunk: Buffer) =>
        chunk.toString().split('\n').filter(s => s.length > 0).forEach(line =>
            console.log(`[python ${id}] stdout: ${line}`)));
    process.stderr.on('data', (chunk: Buffer) =>
        chunk.toString().split('\n').filter(s => s.length > 0).forEach(line =>
            console.log(`[python ${id}] stdout: ${line}`)));

    res.status(200).send();
})

app.listen(port)
