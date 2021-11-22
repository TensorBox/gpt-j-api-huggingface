const axios = require('axios');

const json = JSON.stringify({ prompt: "In a shocking finding, scientist discovered", max_length: 100 });
axios.post('http://0.0.0.0:5000/generate', json, {
  headers: {
    // Overwrite Axios's automatically set Content-Type
    'Content-Type': 'application/json'
  }
}).then(res=>{
    console.log(res.data); // '{"answer":42}'
})
