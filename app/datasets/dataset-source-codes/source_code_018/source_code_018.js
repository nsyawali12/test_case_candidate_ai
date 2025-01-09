const fs = require('fs');
const Joi = require('joi');

// Define your JSON schema using Joi
const schema = Joi.object({
  // Define your schema structure here
  name: Joi.string().required(),
  age: Joi.number().integer().min(18),
  email: Joi.string().email(),
  // Add more properties as needed
});

// Function to read and validate JSON configuration
function validateJsonConfig(filePath) {
  try {
    // Read the JSON configuration file
    const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf8'));

    // Validate against the schema
    const { error } = schema.validate(jsonData);

    if (error) {
      console.error('Validation Error:', error.message);
    } else {
      console.log('Validation Successful. Configuration is valid.');
    }
  } catch (err) {
    console.error('Error reading or parsing the JSON file:', err.message);
  }
}

// Provide the path to your JSON configuration file
const configFilePath = 'path/to/your/config.json';

// Call the function to validate the JSON configuration
validateJsonConfig(configFilePath);