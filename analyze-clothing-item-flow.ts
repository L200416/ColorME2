
'use server';
/**
 * @fileOverview AI-powered clothing item analysis from an image.
 *
 * - analyzeClothingItem - A function that analyzes a clothing item from an image.
 * - AnalyzeClothingItemInput - The input type for the analyzeClothingItem function.
 * - AnalyzeClothingItemOutput - The return type for the analyzeClothingItem function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';
import { ZodError } from 'zod';

const AnalyzeClothingItemInputSchema = z.object({
  photoDataUri: z
    .string()
    .describe(
      "A photo of a clothing item, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
});
export type AnalyzeClothingItemInput = z.infer<typeof AnalyzeClothingItemInputSchema>;

const AnalyzeClothingItemOutputSchema = z.object({
  itemName: z.string().describe('A suggested name for the clothing item. Example: "Blauwe Katoenen T-shirt", "Gouden Ketting"'),
  itemType: z.string().describe("The type of clothing item. For tops, use specific Dutch categories like: 'T-shirt', 'T-shirt met lange mouwen', 'Mouwloos T-shirt', 'Polo', 'Tanktop', 'Hemdje', 'Croptop', 'Blouse', 'Overhemd', 'Sweater', 'Hoodie', 'Trui', 'Spencer', 'Sporttop', 'Body'. For other items, use general Dutch terms like 'Jeans', 'Jurk', 'Jas', 'Broek', 'Rok', 'Schoenen'. For jewelry, prefer specific types like 'Ketting', 'Armband', 'Oorbellen', 'Ring', or use the general term 'Sieraad'. For headwear, prefer 'Pet' or 'Muts', or use the general term 'Hoofddeksel'. For other accessories, use 'Accessoire' or specific terms like 'Tas', 'Riem', 'Sjaal'."),
  itemColor: z.string().describe('The primary color of the clothing item. Example: "Marineblauw", "Goudkleurig"'),
  itemStyle: z.string().describe('The style of the clothing item. Example: "Casual", "Zakelijk", "Bohemian", "Stoer", "Elegant"'),
  fullDescription: z.string().describe('A concise description of the clothing item including its characteristics, suitable for notes.'),
});
export type AnalyzeClothingItemOutput = z.infer<typeof AnalyzeClothingItemOutputSchema>;

export async function analyzeClothingItem(input: AnalyzeClothingItemInput): Promise<AnalyzeClothingItemOutput> {
  return analyzeClothingItemFlow(input);
}

const prompt = ai.definePrompt({
  name: 'analyzeClothingItemPrompt',
  input: {schema: AnalyzeClothingItemInputSchema},
  output: {schema: AnalyzeClothingItemOutputSchema},
  prompt: `You are an expert fashion AI assistant for a Dutch-speaking user. Analyze the provided image of a clothing item.
Identify its key characteristics and provide a suggested name (itemName), type (itemType), color (itemColor), style (itemStyle), and a concise overall description (fullDescription) in Dutch.

For itemType:
- If the item is a top, be specific. Use categories like: 'T-shirt', 'T-shirt met lange mouwen', 'Mouwloos T-shirt', 'Polo', 'Tanktop', 'Hemdje', 'Croptop', 'Blouse', 'Overhemd', 'Sweater', 'Hoodie', 'Trui', 'Spencer', 'Sporttop', 'Body'.
- For other clothing items, use common Dutch terms like 'Jeans', 'Jurk', 'Rok', 'Jas', 'Broek', 'Schoenen'.
- If the item is jewelry, try to be specific: 'Ketting', 'Armband', 'Oorbellen', 'Ring'. If it's unclear, use the general term 'Sieraad'.
- If the item is headwear, try to be specific: 'Pet' (for caps/baseball caps), 'Muts' (for beanies/winter hats). If it's unclear, use the general term 'Hoofddeksel'.
- For other accessories (bags, belts, scarves), use 'Accessoire' or more specific terms like 'Tas', 'Riem', 'Sjaal'.

For itemStyle, use terms like 'Casual', 'Zakelijk', 'Bohemian', 'Stoer', 'Elegant', etc.
The fullDescription should be suitable for a notes field in a digital closet app.

Image: {{media url=photoDataUri}}

Output the analysis in the specified JSON format.`,
});

const analyzeClothingItemFlow = ai.defineFlow(
  {
    name: 'analyzeClothingItemFlow',
    inputSchema: AnalyzeClothingItemInputSchema,
    outputSchema: AnalyzeClothingItemOutputSchema,
  },
  async (input: AnalyzeClothingItemInput): Promise<AnalyzeClothingItemOutput> => {
    const maxRetries = 3;
    let delay = 2000;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const { output } = await prompt(input);

        if (!output) {
          throw new Error('AI-analyse antwoordde, maar de output was leeg of niet in het verwachte formaat.');
        }
        // Validate the output against the Zod schema before returning
        // This explicit validation helps catch mismatches earlier.
        AnalyzeClothingItemOutputSchema.parse(output); 
        return output;
      } catch (error: any) {
        if (error instanceof ZodError) {
          console.error("Zod validation error in analyzeClothingItemFlow:", error.errors);
          const zodErrorMessages = error.errors.map(e => `${e.path.join('.') || 'item'}: ${e.message}`).join('; ');
          throw new Error(
            `AI-output validatiefout: De data van de AI voldoet niet aan het verwachte formaat. Details: ${zodErrorMessages}`
          );
        }

        const errorMessage = String(error.message || 'Onbekende fout opgetreden').toLowerCase();

        const isRetryableError =
          errorMessage.includes('503') ||
          errorMessage.includes('service unavailable') ||
          errorMessage.includes('model is overloaded') ||
          errorMessage.includes('model_is_overloaded') ||
          errorMessage.includes('resource has been exhausted') ||
          errorMessage.includes('rate limit') ||
          errorMessage.includes('try again');

        if (isRetryableError) {
          if (attempt < maxRetries - 1) {
            await new Promise(resolve => setTimeout(resolve, delay));
            delay *= 2;
          } else {
            throw new Error(
              `AI-analyse mislukt na ${maxRetries} pogingen vanwege API-problemen (bijv. overbelasting, rate limits). Laatste fout: ${error.message || 'Onbekende API fout'}`
            );
          }
        } else {
          console.error("Niet-herstelbare fout in analyzeClothingItemFlow:", error);
          throw error; // Re-throw non-retryable or Zod validation errors
        }
      }
    }
    // This line should theoretically be unreachable if errors are handled correctly above
    throw new Error('AI-analyse is onverwacht mislukt na alle pogingen.');
  }
);
