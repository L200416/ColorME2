
'use server';
/**
 * @fileOverview AI-powered outfit suggestion flow.
 *
 * - generateOutfitSuggestion - A function that generates outfit suggestions based on user preferences, closet, and weather.
 * - GenerateOutfitSuggestionInput - The input type for the generateOutfitSuggestion function.
 * - GenerateOutfitSuggestionOutput - The return type for the generateOutfitSuggestion function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const GenerateOutfitSuggestionInputSchema = z.object({
  closetDescription: z
    .string()
    .describe('Description of the user clothes in their digital closet.'),
  weatherCondition: z.string().describe('The current weather condition.'),
  stylePreferences: z.string().describe('The user style preferences.'),
  fashionTrends: z.string().describe('The current fashion trends.'),
});
export type GenerateOutfitSuggestionInput = z.infer<
  typeof GenerateOutfitSuggestionInputSchema
>;

const GenerateOutfitSuggestionOutputSchema = z.object({
  outfitSuggestion: z.string().describe('The generated outfit suggestion text for the main clothing items (top, bottom, outerwear). Shoes and socks will be suggested separately.'),
  reasoning: z.string().describe('The reasoning behind the outfit suggestion.'),
  suggestedShoes: z.string().describe("Specific suggestion for shoes that complement the outfit. E.g., 'Witte sneakers', 'Elegante zwarte pumps'."),
  suggestedSocks: z.string().optional().describe("Specific suggestion for socks, if applicable and visible, that complement the outfit. E.g., 'Onzichtbare sokken', 'Donkerblauwe wollen sokken'."),
  outfitImageUrl: z.string().url().describe("A data URI of the AI-generated image representing the outfit suggestion. Expected format: 'data:image/png;base64,<encoded_data>'."),
});
export type GenerateOutfitSuggestionOutput = z.infer<
  typeof GenerateOutfitSuggestionOutputSchema
>;

export async function generateOutfitSuggestion(
  input: GenerateOutfitSuggestionInput
): Promise<GenerateOutfitSuggestionOutput> {
  return generateOutfitSuggestionFlow(input);
}

const textPrompt = ai.definePrompt({
  name: 'generateOutfitSuggestionTextPrompt',
  input: {schema: GenerateOutfitSuggestionInputSchema},
  output: {schema: z.object({
    outfitSuggestion: z.string().describe('The generated outfit suggestion text for the main clothing items (top, bottom, outerwear). Shoes and socks will be suggested separately.'),
    reasoning: z.string().describe('The reasoning behind the outfit suggestion.'),
    suggestedShoes: z.string().describe("Specific suggestion for shoes that complement the outfit. E.g., 'Witte sneakers', 'Elegante zwarte pumps'."),
    suggestedSocks: z.string().optional().describe("Specific suggestion for socks, if applicable and visible, that complement the outfit. E.g., 'Onzichtbare sokken', 'Donkerblauwe wollen sokken'."),
  })},
  prompt: `You are a personal AI stylist that generates outfit suggestions for users.

  Consider the following information when making your suggestion:

  User Closet: {{{closetDescription}}}
  Weather: {{{weatherCondition}}}
  Style Preferences: {{{stylePreferences}}}
  Fashion Trends: {{{fashionTrends}}}

  Generate a detailed outfit suggestion for the main clothing items (top, bottom, outerwear etc.) and provide a brief reasoning.
  Also, provide a specific suggestion for SHOES that would go well with this outfit.
  If socks are relevant and would be visible or important for the style (e.g., with certain shoes or skirts), also provide a suggestion for SOCKS.
  Format the output in JSON according to the schema. Ensure all text is in Dutch.
  `,
});

const generateOutfitSuggestionFlow = ai.defineFlow(
  {
    name: 'generateOutfitSuggestionFlow',
    inputSchema: GenerateOutfitSuggestionInputSchema,
    outputSchema: GenerateOutfitSuggestionOutputSchema,
  },
  async (input: GenerateOutfitSuggestionInput): Promise<GenerateOutfitSuggestionOutput> => {
    // 1. Generate textual suggestion, reasoning, shoes, and socks
    const {output: textOutput} = await textPrompt(input);
    if (!textOutput || !textOutput.outfitSuggestion || !textOutput.suggestedShoes || !textOutput.reasoning) {
      throw new Error('AI response for outfit text suggestion was empty, incomplete, or not in the expected format.');
    }

    // 2. Generate image based on the textual outfit suggestion including shoes and socks
    const imagePromptText = `Generate a high-quality, realistic image of a person wearing the following outfit: ${textOutput.outfitSuggestion}. The person should also be wearing ${textOutput.suggestedShoes}${textOutput.suggestedSocks ? ' with ' + textOutput.suggestedSocks : ''}. Ensure the style is fashionable and clear, and that the shoes and any visible socks are clearly depicted. The image should be suitable for a fashion app.`;
    
    const {media} = await ai.generate({
      model: 'googleai/gemini-2.0-flash-exp',
      prompt: imagePromptText,
      config: {
        responseModalities: ['TEXT', 'IMAGE'],
      },
    });

    if (!media?.url) {
      throw new Error('Failed to generate outfit image.');
    }

    return {
      outfitSuggestion: textOutput.outfitSuggestion,
      reasoning: textOutput.reasoning,
      suggestedShoes: textOutput.suggestedShoes,
      suggestedSocks: textOutput.suggestedSocks,
      outfitImageUrl: media.url,
    };
  }
);
