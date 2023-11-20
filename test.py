import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from docx import Document
from docx.shared import Inches

class PhotoViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Viewer")

        # Variables to store user information
        self.name_var = tk.StringVar()
        self.surname_var = tk.StringVar()
        self.agency_var = tk.StringVar()
        self.links_var = tk.StringVar()
        self.image_path = None  # To store the path of the loaded image

        # Image display label
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # Entry fields for user information
        name_label = tk.Label(root, text="Name:")
        name_label.pack()
        name_entry = tk.Entry(root, textvariable=self.name_var)
        name_entry.pack()

        surname_label = tk.Label(root, text="Surname:")
        surname_label.pack()
        surname_entry = tk.Entry(root, textvariable=self.surname_var)
        surname_entry.pack()

        agency_label = tk.Label(root, text="Agency:")
        agency_label.pack()
        agency_entry = tk.Entry(root, textvariable=self.agency_var)
        agency_entry.pack()

        links_label = tk.Label(root, text="Links (comma-separated):")
        links_label.pack()
        links_entry = tk.Entry(root, textvariable=self.links_var)
        links_entry.pack()

        # Button to load photo
        load_button = tk.Button(root, text="Load Photo", command=self.load_photo)
        load_button.pack(pady=10)

        # Button to save information to Word file
        save_button = tk.Button(root, text="Save to Word", command=self.save_to_word)
        save_button.pack(pady=10)

    def load_photo(self):
        initial_dir = os.path.expanduser("~")  # Set the initial directory to the user's home directory
        file_path = filedialog.askopenfilename(
            title="Select a Photo",
            initialdir=initial_dir
        )

        if file_path:
            try:
                image = Image.open(file_path)
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo
                self.image_path = file_path
            except Exception as e:
                print(f"Error loading photo: {e}")
        else:
            print("No file selected.")

    def save_to_word(self):
        if not self.image_path:
            print("Please load a photo before saving to Word.")
            return

        document = Document()
        document.add_heading('User Information', level=1)

        # Add image to the Word document
        document.add_picture(self.image_path, width=Inches(2.0))

        # Create a table
        table = document.add_table(rows=1, cols=4)
        table.autofit = False
        table.columns[0].width = Inches(1.0)
        table.columns[1].width = Inches(1.5)
        table.columns[2].width = Inches(1.5)
        table.columns[3].width = Inches(2.0)

        # Set the header row
        header_cells = table.rows[0].cells
        header_cells[0].text = "Name"
        header_cells[1].text = "Surname"
        header_cells[2].text = "Agency"
        header_cells[3].text = "Links"

        # Add user information to the table
        row_cells = table.add_row().cells
        row_cells[0].text = self.name_var.get()
        row_cells[1].text = self.surname_var.get()
        row_cells[2].text = self.agency_var.get()

        links = [link.strip() for link in self.links_var.get().split(',')]
        links_text = "\n".join(links)
        row_cells[3].text = links_text

        word_file_path = filedialog.asksaveasfilename(
            title="Save Word File",
            defaultextension=".docx",
            filetypes=[("Word files", "*.docx")]
        )

        if word_file_path:
            document.save(word_file_path)
            print(f"Information saved to: {word_file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoViewer(root)
    root.mainloop()
