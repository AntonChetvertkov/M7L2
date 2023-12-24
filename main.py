import discord
from discord.ext import commands
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

np.set_printoptions(suppress=True)

def predict_image_class(image_path, model_path, labels_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_path)
    labels_path = os.path.join(script_dir, labels_path)

    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r", encoding="utf-8").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='^', intents=discord.Intents.default())

@bot.event
async def on_ready():
    print(f'I {bot.user}')

@bot.command()
async def who(ctx):
    await ctx.send(f'I am Antons bot')

@bot.command

@bot.command()
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            file_url = attachment.url
            await attachment.save(f'M7/{attachment.filename}')
            result = predict_image_class(model_path="./keras_model.h5", labels_path="./labels.txt", image_path=f"./M7/{attachment.filename}")
            await ctx.send(f"Prediction: {result[0]}, Confidence: {result[1]:.4f}")
    else:
        await ctx.send('No image = no output')

bot.run('token')
