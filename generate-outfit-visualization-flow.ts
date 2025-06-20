
'use server';
/**
 * @fileOverview AI-powered outfit visualization flow.
 *
 * - generateOutfitVisualization - A function that generates an image of an outfit.
 * - GenerateOutfitVisualizationInput - The input type for the generateOutfitVisualization function.
 * - GenerateOutfitVisualizationOutput - The return type for the generateOutfitVisualization function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const ClothingItemVisualSchema = z.object({
  description: z.string().describe('A short description of the clothing item (e.g., "Blue denim jacket", "White cotton t-shirt").'),
  imageUrl: z.string().url().optional().describe("Optional: A data URI of the item's image for better visual reference by the AI. Expected format: 'data:<mimetype>;base64,<encoded_data>'."),
});

const GenerateOutfitVisualizationInputSchema = z.object({
  description: z.string().describe('A detailed textual description of the complete outfit to be visualized, combining all items.'),
  items: z.array(ClothingItemVisualSchema).optional().describe('Optional: An array of individual clothing items with their descriptions and image URIs to provide more context to the AI.'),
});
export type GenerateOutfitVisualizationInput = z.infer<typeof GenerateOutfitVisualizationInputSchema>;

const GenerateOutfitVisualizationOutputSchema = z.object({
  visualizationUrl: z.string().url().describe("A data URI of the AI-generated image representing the outfit. Expected format: 'data:image/png;base64,<encoded_data>'."),
});
export type GenerateOutfitVisualizationOutput = z.infer<typeof GenerateOutfitVisualizationOutputSchema>;

export async function generateOutfitVisualization(
  input: GenerateOutfitVisualizationInput
): Promise<GenerateOutfitVisualizationOutput> {
  return generateOutfitVisualizationFlow(input);
}

const generateOutfitVisualizationFlow = ai.defineFlow(
  {
    name: 'generateOutfitVisualizationFlow',
    inputSchema: GenerateOutfitVisualizationInputSchema,
    outputSchema: GenerateOutfitVisualizationOutputSchema,
  },
  async (input: GenerateOutfitVisualizationInput): Promise<GenerateOutfitVisualizationOutput> => {
    let promptSegments = [];
    promptSegments.push({text: `Generate a high-quality, realistic image of a person wearing the following outfit: ${input.description}.`});
    promptSegments.push({text: "Ensure the style is fashionable and clear. The image should be suitable for a fashion app."});

    if (input.items && input.items.length > 0) {
      promptSegments.push({text: "\n\nFor additional context, here are some of the items included (prioritize the main description above):"});
      for (const item of input.items) {
        if (item.imageUrl) {
          // Ensure the image URL is a valid data URI before passing to the model
          if (item.imageUrl.startsWith('data:image')) {
            promptSegments.push({media: {url: item.imageUrl}});
            promptSegments.push({text: `This is a ${item.description}.`});
          } else {
            console.warn(`Skipping invalid image URL for visualization: ${item.imageUrl}`);
            promptSegments.push({text: `(Image for ${item.description} was not in correct data URI format)`});
          }
        } else {
          promptSegments.push({text: `- ${item.description}`});
        }
      }
    }
    
    const {media} = await ai.generate({
      model: 'googleai/gemini-2.0-flash-exp', // Using the experimental model that supports image generation
      prompt: promptSegments,
      config: {
        responseModalities: ['TEXT', 'IMAGE'], // Must request TEXT even if only IMAGE is primary
      },
       safetySettings: [ // Adjust safety settings if needed
        { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_LOW_AND_ABOVE' },
        { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
        { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE'},
      ],
    });

    if (!media?.url) {
      throw new Error('Failed to generate outfit visualization image. The AI might have declined the request or an unknown error occurred.');
    }

    return {
      visualizationUrl: media.url,
    };
  }
);
