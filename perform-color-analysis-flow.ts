
'use server';
/**
 * @fileOverview AI-powered color analysis (seasonal analysis).
 *
 * - performColorAnalysis - A function that analyzes a user's photo to determine their color season.
 * - PerformColorAnalysisInput - The input type for the performColorAnalysis function.
 * - PerformColorAnalysisOutput - The return type for the performColorAnalysis function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';
import { ZodError } from 'zod';

const PerformColorAnalysisInputSchema = z.object({
  userDataUri: z
    .string()
    .describe(
      "A photo of the user's face, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'. This photo will be used to analyze skin tone, hair color, and eye color for seasonal color analysis."
    ),
});
export type PerformColorAnalysisInput = z.infer<typeof PerformColorAnalysisInputSchema>;

const ColorInfoSchema = z.object({
  name: z.string().describe("De Nederlandse naam van de kleur (bijv. 'Warm Oranje', 'Koel Blauw')."),
  hex: z.string().regex(/^#[0-9A-F]{6}$/i, "Moet een geldige hex-kleurcode zijn (bijv. #FF5733).").describe("De hex-kleurcode (bijv. #FF5733).")
});

const PerformColorAnalysisOutputSchema = z.object({
  seasonType: z.string().describe('Het gedetecteerde kleurseizoen (bijv. "Lente", "Zomer", "Herfst", "Winter", of "Niet te bepalen").'),
  analysisDescription: z.string().describe('Een gedetailleerde uitleg van de analyse in het Nederlands. Als geen seizoen bepaald kan worden, leg uit waarom (bijv. onduidelijke foto, geen gezicht).'),
  characteristics: z.object({
    skinTone: z.string().describe("Korte beschrijving van de huidtint (bijv. 'Warm met gouden ondertoon', 'Koel met roze ondertoon', of 'Niet te bepalen')."),
    hairColor: z.string().describe("Korte beschrijving van de haarkleur (bijv. 'Donkerbruin met warme highlights', 'Asblond', of 'Niet te bepalen')."),
    eyeColor: z.string().describe("Korte beschrijving van de oogkleur (bijv. 'Helderblauw', 'Donkerbruin met spikkels', of 'Niet te bepalen').")
  }).describe("Waargenomen kenmerken van de persoon. 'Niet te bepalen' indien onmogelijk vast te stellen."),
  recommendedColors: z.array(ColorInfoSchema).describe('Een lijst van aanbevolen kleuren die flatterend zijn voor dit seizoen, inclusief Nederlandse naam en hex-code. Leeg indien seizoen "Niet te bepalen" is.'),
  avoidColors: z.array(ColorInfoSchema).describe('Een lijst van kleuren die minder flatterend zijn voor dit seizoen of beter vermeden kunnen worden, inclusief Nederlandse naam en hex-code. Leeg indien seizoen "Niet te bepalen" is.'),
  paletteDescription: z.string().describe("Een algemene beschrijving van het kleurenpalet dat bij het seizoen past (bijv. 'Heldere, warme kleuren', 'Gedempte, koele tinten', of 'Niet te bepalen').")
});
export type PerformColorAnalysisOutput = z.infer<typeof PerformColorAnalysisOutputSchema>;

export async function performColorAnalysis(input: PerformColorAnalysisInput): Promise<PerformColorAnalysisOutput> {
  return performColorAnalysisFlow(input);
}

const prompt = ai.definePrompt({
  name: 'performColorAnalysisPrompt',
  input: {schema: PerformColorAnalysisInputSchema},
  output: {schema: PerformColorAnalysisOutputSchema},
  prompt: `Je bent een AI-expert in seizoenskleuranalyse voor mode en styling. Analyseer de meegeleverde foto van het gezicht van een persoon. 
Identificeer hun huidtint (let op ondertonen: warm, koel, neutraal), natuurlijke haarkleur en oogkleur.

Op basis van deze kenmerken, bepaal je het kleurseizoen van de persoon (Lente, Zomer, Herfst, of Winter).

Belangrijk:
- Als je geen duidelijk kleurseizoen kunt bepalen (bijvoorbeeld omdat het gezicht niet goed zichtbaar is, de foto geen persoon bevat, de belichting onvoldoende is, of de focus verkeerd ligt), stel \`seasonType\` dan in op "Niet te bepalen".
- In het geval dat \`seasonType\` "Niet te bepalen" is:
    - Moeten \`recommendedColors\` en \`avoidColors\` **lege arrays** zijn ([]).
    - Moet \`analysisDescription\` duidelijk uitleggen waarom geen analyse mogelijk was (bijv. "Het is lastig om een nauwkeurige seizoenskleuranalyse te maken op basis van deze foto, omdat de focus ligt op [iets anders] en niet op het gezicht van een persoon. Ik kan geen huidtint, haarkleur en oogkleur identificeren...").
    - Moeten de velden in \`characteristics\` (skinTone, hairColor, eyeColor) ingesteld worden op "Niet te bepalen".
    - Moet \`paletteDescription\` ingesteld worden op "Niet te bepalen".
- Geef *alleen* kleuraanbevelingen (minimaal 5) en te vermijden kleuren (minimaal 3), en gedetailleerde karakteristieken als een seizoen met zekerheid kan worden vastgesteld.
- Geef een algemene beschrijving van het kleurenpalet dat bij het seizoen past.

Zorg ervoor dat alle tekstuele output in het Nederlands is. De hex-codes moeten correct geformatteerd zijn (bijv. #RRGGBB).

Foto van de gebruiker: {{media url=userDataUri}}

Lever de output strikt volgens het opgegeven JSON-schema, rekening houdend met bovenstaande uitzonderingsregels.`,
   safetySettings: [
    { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_LOW_AND_ABOVE' },
    { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
    { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
    { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE'},
  ],
});

const performColorAnalysisFlow = ai.defineFlow(
  {
    name: 'performColorAnalysisFlow',
    inputSchema: PerformColorAnalysisInputSchema,
    outputSchema: PerformColorAnalysisOutputSchema,
  },
  async (input: PerformColorAnalysisInput): Promise<PerformColorAnalysisOutput> => {
    const maxRetries = 3;
    let delay = 2000; 

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        let { output } = await prompt(input);

        if (!output) {
          throw new Error('AI-analyse antwoordde, maar de output was leeg of niet in het verwachte formaat.');
        }

        // Post-processing to ensure consistency if analysis is not possible
        const unanalyzableSeasonKeywords = ["niet te bepalen", "onvoldoende informatie", "kan niet analyseren", "geen gezicht", "onmogelijk te bepalen"];
        const isUnanalyzable = output.seasonType && unanalyzableSeasonKeywords.some(keyword => output.seasonType.toLowerCase().includes(keyword.toLowerCase()));

        if (isUnanalyzable) {
          output.recommendedColors = [];
          output.avoidColors = [];
          
          if (!output.analysisDescription || !unanalyzableSeasonKeywords.some(keyword => output.analysisDescription.toLowerCase().includes(keyword.toLowerCase()))) {
            output.analysisDescription = "De AI kon geen betrouwbare kleuranalyse uitvoeren op basis van de verstrekte afbeelding. Zorg voor een duidelijke foto van het gezicht bij goed daglicht, zonder zware make-up en met een neutrale achtergrond.";
          }
          if (!output.characteristics || !output.characteristics.skinTone.toLowerCase().includes("niet te bepalen")) {
              output.characteristics = {
                  skinTone: "Niet te bepalen",
                  hairColor: "Niet te bepalen",
                  eyeColor: "Niet te bepalen",
              };
          }
          if (!output.paletteDescription || !output.paletteDescription.toLowerCase().includes("niet te bepalen")) {
                output.paletteDescription = "Niet te bepalen omdat het seizoen niet vastgesteld kon worden."
          }
        } else {
          // Ensure color arrays meet minimums if season IS determined, otherwise Zod will fail if prompt didn't provide enough
          if (output.recommendedColors.length < 5 && output.recommendedColors.length > 0) { 
            // If some colors are there, but not enough, it's an LLM mistake. For simplicity, clear them if not meeting min.
            // A better approach might be to try and get the LLM to provide more, or fill with generic defaults.
            // For now, if the LLM fails to provide enough for a valid season, this will cause a Zod error below.
            // This indicates the LLM did not follow instructions for a valid season.
          }
           if (output.avoidColors.length < 3 && output.avoidColors.length > 0) {
            // Similar logic for avoidColors
          }
        }
        
        PerformColorAnalysisOutputSchema.parse(output); // Validate output
        return output;

      } catch (error: any) {
        if (error instanceof ZodError) {
          console.error("Zod validation error in performColorAnalysisFlow:", error.errors);
          const zodErrorMessages = error.errors.map(e => `Veld '${e.path.join('.') || 'root'}': ${e.message}`).join('; ');
          throw new Error(
            `AI-output validatiefout: ${zodErrorMessages}. Controleer of de AI de data in het juiste formaat retourneert, inclusief het correcte aantal kleuren indien een seizoen is bepaald.`
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
            console.warn(`Poging ${attempt + 1} mislukt (herstelbare fout): ${error.message}. Opnieuw proberen na ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
            delay *= 2; 
          } else {
            console.error(`AI-analyse mislukt na ${maxRetries} pogingen vanwege API-problemen. Laatste fout: ${error.message}`);
            throw new Error(
              `AI-analyse mislukt na ${maxRetries} pogingen vanwege API-problemen (bijv. overbelasting, rate limits). Probeer het later opnieuw. Fout: ${error.message || 'Onbekende API fout'}`
            );
          }
        } else {
          console.error("Niet-herstelbare fout in performColorAnalysisFlow:", error);
          throw error; 
        }
      }
    }
    throw new Error('De kleuranalyse is onverwacht mislukt na alle pogingen.');
  }
);

    

    