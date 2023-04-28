import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Dada a conversa a seguir e uma pergunta de acompanhamento, reformule a pergunta de acompanhamento para ser uma pergunta independente.

Histórico de conversa:
{chat_history}
Entrada de acompanhamento: {question}
Pergunta autônoma:`;

// const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
// If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
// If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

// {context}

// Question: {question}
// Helpful answer in markdown:`;

const QA_PROMPT = `Você é um assistente de IA que fornece conselhos úteis. Você recebe as seguintes partes extraídas de um longo documento e uma pergunta. Forneça uma resposta de conversação com base no contexto fornecido.
Você só deve fornecer hiperlinks que façam referência ao contexto abaixo. NÃO crie hiperlinks.
Se você não conseguir encontrar a resposta no contexto abaixo, diga "Hmm, não tenho certeza". Não tente inventar uma resposta.
Se a pergunta não estiver relacionada ao contexto, responda educadamente que você está sintonizado para responder apenas perguntas relacionadas ao contexto.

{context}

Pergunta: {question}
Resposta útil em markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-4', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
