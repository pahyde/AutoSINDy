const express = require('express')
const fs = require('fs').promises
const cors = require('cors')

const app = express()
const PORT = process.env.PORT || 5000

app.use(express.json())
app.use(cors({
    origin: '*'
}))

const readFile = async path => {
    const data = await fs.readFile(path, 'utf-8')
    return data
}

app.get('/favicon.ico', (req, res) => res.status(204));


app.get("/:filename", async (req, res) => {
    res.set('content-type', 'application/pdf')
    let filename
    try {
        filename = req.params.filename
    } catch (err) {
        console.error(err)
        res.send({err: 'file error'})
    }
    const dir = '../../final-paper/build/'
    const file = await readFile(`${dir}/${filename}`)
    res.send(file)
})


app.listen(PORT, () => {
    console.log('Colab Server listening on port: ', PORT)
})

