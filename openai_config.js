/**
 * OpenAI Integration Test for Lucky Train AI Assistant
 * 
 * This file tests the OpenAI integration with the provided API keys
 */

require('dotenv').config();
const { OpenAI } = require('openai');

// Initialize OpenAI client with environment variables
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    organization: process.env.OPENAI_ORGANIZATION_ID
});

// Test OpenAI connection
async function testOpenAIConnection() {
    try {
        const response = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: "You are a helpful assistant." },
                { role: "user", content: "Hello!" }
            ],
            temperature: 0.7,
        });
        console.log("OpenAI API connection successful!");
        console.log("Response:", response.choices[0].message.content);
    } catch (error) {
        console.error("Error connecting to OpenAI API:", error.message);
    }
}

// Export functions for use in the main application
module.exports = {
    openai,
    testOpenAIConnection
};

// If this file is run directly, test the connection
if (require.main === module) {
    testOpenAIConnection();
} 