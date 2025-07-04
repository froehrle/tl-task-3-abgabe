import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.50.2';
const openAIApiKey = Deno.env.get('OPENAI_API_KEY');
const supabaseUrl = Deno.env.get('SUPABASE_URL');
const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type'
};
const supabase = createClient(supabaseUrl, supabaseServiceKey);
serve(async (req)=>{
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      headers: corsHeaders
    });
  }
  try {
    const { conversationHistory, courseId, teacherId } = await req.json();
    console.log('Generate questions from chat request:', {
      courseId,
      teacherId
    });
    // Define smart defaults
    const defaults = {
      schwierigkeitsgrad: 'mittel',
      anzahl_fragen: 5,
      thema: 'Allgemeine Fragen',
      fragetyp: 'Verständnisfragen',
      zielgruppe: 'Studenten',
      keywords: ''
    };
    // Get the latest user message for context-aware extraction
    const userMessages = conversationHistory.filter((msg)=>msg.role === 'user');
    const latestUserMessage = userMessages[userMessages.length - 1]?.content || '';
    console.log('Latest user message for parameter extraction:', latestUserMessage);
    // Extract parameters from the latest user message using OpenAI
    const extractionPrompt = `Du bist ein Parameter-Extractor für ein Fragenerstellungs-System. 

WICHTIG: Analysiere NUR die LETZTE Benutzeranfrage, nicht das gesamte Gespräch.

Letzte Benutzeranfrage: "${latestUserMessage}"

Standardwerte:
- schwierigkeitsgrad: "mittel"
- anzahl_fragen: 5
- thema: "Allgemeine Fragen"
- fragetyp: "Verständnisfragen"
- zielgruppe: "Studenten"
- keywords: ""

SPEZIELLE REGELN für "mehr" oder "zusätzliche" Fragen:
- Wenn der Benutzer "X mehr", "X zusätzliche", "weitere X" sagt, dann anzahl_fragen = X
- Beispiele: "3 mehr" → anzahl_fragen: 3, "2 zusätzliche" → anzahl_fragen: 2

Extrahiere nur Parameter, die explizit in der letzten Nachricht erwähnt werden. Verwende Standardwerte für alle anderen.

Antworte nur mit JSON:
{
  "schwierigkeitsgrad": "leicht" | "mittel" | "schwer",
  "anzahl_fragen": number (1-20),
  "thema": "string",
  "fragetyp": "Verständnisfragen" | "Rechenfragen",
  "zielgruppe": "string",
  "keywords": "string"
}`;
    console.log('Sending extraction request to OpenAI...');
    const extractionResponse = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openAIApiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'user',
            content: extractionPrompt
          }
        ],
        temperature: 0.1,
        max_tokens: 500
      })
    });
    if (!extractionResponse.ok) {
      console.error('OpenAI extraction failed:', extractionResponse.status, await extractionResponse.text());
      throw new Error(`OpenAI extraction failed: ${extractionResponse.status}`);
    }
    const extractionData = await extractionResponse.json();
    let extractedParams;
    try {
      const extractedContent = extractionData.choices[0].message.content;
      console.log('OpenAI extraction response:', extractedContent);
      extractedParams = JSON.parse(extractedContent);
      // Merge with defaults to ensure all required fields are present
      extractedParams = {
        ...defaults,
        ...extractedParams
      };
      // Validate extracted parameters
      if (![
        'leicht',
        'mittel',
        'schwer'
      ].includes(extractedParams.schwierigkeitsgrad)) {
        console.warn('Invalid schwierigkeitsgrad, using default:', extractedParams.schwierigkeitsgrad);
        extractedParams.schwierigkeitsgrad = defaults.schwierigkeitsgrad;
      }
      if (![
        'Verständnisfragen',
        'Rechenfragen'
      ].includes(extractedParams.fragetyp)) {
        console.warn('Invalid fragetyp, using default:', extractedParams.fragetyp);
        extractedParams.fragetyp = defaults.fragetyp;
      }
      if (extractedParams.anzahl_fragen < 1 || extractedParams.anzahl_fragen > 20) {
        console.warn('Invalid anzahl_fragen, using default:', extractedParams.anzahl_fragen);
        extractedParams.anzahl_fragen = defaults.anzahl_fragen;
      }
    } catch (parseError) {
      console.error('Failed to parse extracted parameters, using defaults:', parseError);
      console.error('Raw OpenAI response:', extractionData);
      extractedParams = defaults;
    }
    console.log('Final parameters for Lambda:', extractedParams);
    // Call the existing lambda function with extracted parameters
    console.log('Calling Lambda function with extracted parameters:', extractedParams);
    const lambdaResponse = await fetch('https://o662virii4xhey5nodl4n5umi40uuddp.lambda-url.eu-central-1.on.aws/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(extractedParams)
    });
    console.log('Lambda response status:', lambdaResponse.status);
    console.log('Lambda response headers:', Object.fromEntries(lambdaResponse.headers.entries()));
    if (!lambdaResponse.ok) {
      const errorBody = await lambdaResponse.text();
      console.error('Lambda function error details:');
      console.error('Status:', lambdaResponse.status);
      console.error('Status Text:', lambdaResponse.statusText);
      console.error('Response Body:', errorBody);
      console.error('Request Parameters:', JSON.stringify(extractedParams, null, 2));
      throw new Error(`Lambda function error: ${lambdaResponse.status} - ${errorBody}`);
    }
    const lambdaResult = await lambdaResponse.json();
    console.log('Lambda response received successfully:', {
      questionCount: lambdaResult.questions?.questions?.length || lambdaResult.questions?.length || 0,
      fullResponse: JSON.stringify(lambdaResult, null, 2)
    });
    // Handle both nested and flat question response structures
    let questions = [];
    if (Array.isArray(lambdaResult.questions)) {
      questions = lambdaResult.questions;
    } else if (lambdaResult.questions?.questions && Array.isArray(lambdaResult.questions.questions)) {
      questions = lambdaResult.questions.questions;
    }
    if (questions.length === 0) {
      console.warn('No questions returned from Lambda function');
      throw new Error('Lambda function returned no questions');
    }
    // Save questions to pending_questions table
    const pendingQuestions = questions.map((q)=>({
        course_id: courseId,
        teacher_id: teacherId,
        question_text: q.question_text,
        question_type: q.question_type || 'multiple_choice',
        question_style: extractedParams.fragetyp,
        options: q.options || q.multiple_choice_options,
        correct_answer: q.correct_answer,
        chat_context: conversationHistory,
        status: 'pending'
      }));
    const { data: insertedQuestions, error: insertError } = await supabase.from('pending_questions').insert(pendingQuestions).select();
    if (insertError) {
      console.error('Error inserting pending questions:', insertError);
      throw new Error('Failed to save questions');
    }
    console.log('Questions saved to pending_questions table:', insertedQuestions?.length);
    return new Response(JSON.stringify({
      success: true,
      questionsGenerated: questions.length,
      extractedParams,
      pendingQuestions: insertedQuestions
    }), {
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      }
    });
  } catch (error) {
    console.error('Error in generate-questions-from-chat function:', error);
    return new Response(JSON.stringify({
      error: error.message,
      success: false
    }), {
      status: 500,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      }
    });
  }
});
