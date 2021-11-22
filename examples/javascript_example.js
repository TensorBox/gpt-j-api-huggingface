const axios = require('axios');

const json = JSON.stringify({ prompt: "In a shocking finding, scientist discovered", max_length: 100 });
const res = await axios.post('http://0.0.0.0:5000/generate', json, {
  headers: {
    // Overwrite Axios's automatically set Content-Type
    'Content-Type': 'application/json'
  }
});

console.log(res.data.data); // '{"answer":42}'