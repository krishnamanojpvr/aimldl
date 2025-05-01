from langchain.docstore.document import Document as langchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

myData = {
    "about": [
        "Meet Galaxy S24 Ultra, the ultimate form of Galaxy Ultra with a new titanium exterior and a 17.25cm (6.8\") flat display. It's an absolute marvel of design.",
        "The legacy of Galaxy Note is alive and well. Write, tap and navigate with precision your fingers wish they had on the new, flat display.",
        "With the most megapixels on a smartphone and AI processing, Galaxy S24 Ultra sets the industry standard for image quality every time you hit the shutter. What's more, the new ProVisual engine recognizes objects — improving colour tone, reducing noise and bringing out detail.",
        "A new way to search is here with Circle to Search. While scrolling your fav social network, use your S Pen or finger to circle something and get Google Search results.",
        "Victory can be yours with the new Snapdragon 8 Gen 3 for Galaxy. Faster processing gives you the power you need for all the gameplay you want. Then, manifest graphic effects in real time with ray tracing for hyper-realistic shadows and reflections."
    ],
    "product_details": {
        "image": "https://m.media-amazon.com/images/I/711zU2HCovL._SX569_.jpg",
        "name": "Samsung Galaxy S24 Ultra 5G AI Smartphone (Titanium Black, 12GB, 256GB Storage)",
        "price": "93,990.",
        "rating": "4.0 out of 5 stars"
    },
    "reviews": [
        "Simply Superb and Awesome..!!!\nS24 Ultra is the best Andriod available out there. Overall Styling, Top notch Camera, Premium feel, Titanium body, Super Advanced Processor, 12GB RAM, Vast Storage capacity (starting from 256GB), Best in class Battery backup, Reliable brand are the key USP's one should consider in this phone. I use IP15 Pro as my Primary phone and after this in hand, honestly iPhone 15 Pro feels outdated..!! Guys, if you are thinking of this phone, then just go for it..! No second thoughts.\n",
        "I recently upgraded to the Samsung Galaxy S24 Ultra, and I couldn't be happier! From the moment I unboxed it, the sleek design and premium feel set a new standard. The screen is stunning – bright, vibrant, and incredibly responsive, making streaming and gaming a real pleasure.\n\nPerformance-wise, this phone is a beast. Multitasking is a breeze, and apps open instantly, thanks to the top-tier processor. Battery life is also fantastic, easily lasting me a full day with heavy use, and fast charging means I'm never stuck waiting long when I need a boost.\n\nThe camera quality is next-level! Low-light photos come out crisp, the zoom is mind-blowing, and colors are captured beautifully. It's like having a professional camera in my pocket.\n\nOverall, the Galaxy S24 Ultra is worth every penny. It's a true flagship device that feels future-proof, and I'd highly recommend it to anyone looking for a powerful, feature-packed smartphone!\n",
        "The phone itself myt be one of thr best phones I have used till date but the problem is that the phone I received is of a different region(software) and not India so it's got defective issues. On contacting amazon service center and samsung service center I keep getting no solutions to my problem and will not allow a replacement since it's a software issue that cannot be solved. Samsung service center keeps holding my phone till the validity of return period and also saying they won't be able to give me the documents to return phone via amazon since their validity of the giving the documents have ended as it was counted from invoice date (date of ordering ) I received the phone after 5 days and have contacted both service centres within 2 days. I'm pretty much helpless now.\n",
        "Best android phone in the market at present.Awesome built and superb camera quality .Happy with the purchase.\n",
        "Excellent phone the camera quality is exceptional. Sound quality and battery life is good. Charging speed is fast.\n",
        "Amazing display, Great cam, charging speed, always cool.\nCons:\nIt doesn't maintain a steady video frame rate in low light.\nWireless 15w charging is slow\n\nMy rant on Apple\nI was interested to get the Apple watch but was hesitant to get the iphone but finally gave in and... IT WAS HORROR...\n\nWorst part, during exchange, you don't have access to your iphone so say bye bye to your WhatsApp backup... I had to get an old iphone from a friend to get the data transfered...\n\nAll that aside, S24 U has been just the right device for me...\n",
        "Very good phone,display and camera superb, smoth processor\n",
        "Works really well and good picture quality and Great zoom quality as well.\n"
    ]
}

def split_documents(chunk_size, raw_knowledge_base, tokenizer_name):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=[
            "\n#{1,6}", "```\n", "\n\\\\\\*+\n", "\n---+\n", "\n_+\n", "\n\n", "\n", " ", ""
        ],
    )
    docs_processed = []
    for doc in raw_knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    return docs_processed_unique

# Create knowledge base
RAW_KNOWLEDGE_BASE = [langchainDocument(page_content=about) for about in myData["about"]]
RAW_KNOWLEDGE_BASE += [langchainDocument(page_content=review) for review in myData["reviews"]]

# Process documents
docs_processed = split_documents(512, RAW_KNOWLEDGE_BASE, "thenlper/gte-small")

# Create embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="thenlper/gte-small", 
    encode_kwargs={"normalize_embeddings": True} # this normalizes the embeddings between -1 and 1. This is done to use cosine distance similariity search
)
KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, 
    embedding_model, 
    distance_strategy=DistanceStrategy.COSINE
)

print(KNOWLEDGE_VECTOR_DATABASE)

# Example query
# user_query = "What is the camera quality of the phone?"
# results = KNOWLEDGE_VECTOR_DATABASE.similarity_search(user_query, k=3)
# for doc in results:
#     print(doc.page_content)
#     print("---")

# all_docs = KNOWLEDGE_VECTOR_DATABASE.docstore._dict.values()
# print(f"Total documents: {len(all_docs)}")
# print(KNOWLEDGE_VECTOR_DATABASE.index)
# print(KNOWLEDGE_VECTOR_DATABASE.distance_strategy)
# print(KNOWLEDGE_VECTOR_DATABASE.embedding_function)











