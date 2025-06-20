// This is an AI-powered function that generates outfit inspiration based on user preferences.
'use server';

/**
 * @fileOverview Outfit inspiration generator based on user preferences.
 *
 * - generateOutfitInspiration - A function that generates outfit inspiration.
 * - GenerateOutfitInspirationInput - The input type for the generateOutfitInspiration function.
 * - GenerateOutfitInspirationOutput - The return type for the generateOutfitInspiration function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const GenerateOutfitInspirationInputSchema = z.object({
  bodyType: z
    .string()
    .describe('The users body type, e.g., apple, pear, hourglass, rectangle.'),
  stylePreferences: z
    .string()
    .describe('The users style preferences, e.g., casual, formal, bohemian.'),
  clothingItems: z
    .array(z.string())
    .describe('A list of clothing items the user owns.'),
});
export type GenerateOutfitInspirationInput = z.infer<
  typeof GenerateOutfitInspirationInputSchema
>;

const GenerateOutfitInspirationOutputSchema = z.object({
  inspiredOutfits: z
    .array(z.string())
    .describe('A list of outfit suggestions based on similar users.'),
});

export type GenerateOutfitInspirationOutput = z.infer<
  typeof GenerateOutfitInspirationOutputSchema
>;

export async function generateOutfitInspiration(
  input: GenerateOutfitInspirationInput
): Promise<GenerateOutfitInspirationOutput> {
  return generateOutfitInspirationFlow(input);
}

const prompt = ai.definePrompt({
  name: 'generateOutfitInspirationPrompt',
  input: {schema: GenerateOutfitInspirationInputSchema},
  output: {schema: GenerateOutfitInspirationOutputSchema},
  prompt: `You are a personal stylist that provides outfit inspiration to users.

You will generate outfit suggestions based on the user's body type, style preferences, and existing clothing items.

Consider outfits worn by other users with similar body types and styles.

Body Type: {{{bodyType}}}
Style Preferences: {{{stylePreferences}}}
Clothing Items: {{#each clothingItems}}{{{this}}}, {{/each}}`,
});

const generateOutfitInspirationFlow = ai.defineFlow(
  {
    name: 'generateOutfitInspirationFlow',
    inputSchema: GenerateOutfitInspirationInputSchema,
    outputSchema: GenerateOutfitInspirationOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    if (!output || !output.inspiredOutfits) {
      throw new Error(
        'AI response for outfit inspiration was empty or not in the expected format.'
      );
    }
    return output;
  }
);
