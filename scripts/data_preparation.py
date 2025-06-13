class DocumentClassificationDataset(Dataset):

    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):

        image_path = self.image_paths[item]
        json_path = image_path.with_suffix(".json")
        with json_path.open("r") as f:
            ocr_result = json.load(f)

            with Image.open(image_path).convert("RGB") as image:

                width, height = image.size
                width_scale = 1000 / width
                height_scale = 1000 / height

                words = []
                boxes = []
                for row in ocr_result:
                    boxes.append(
                        scale_bounding_box(
                            row["bounding_box"], width_scale, height_scale
                        )
                    )
                    words.append(row["word"])

                encoding = self.processor(
                    image,
                    words,
                    boxes=boxes,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

        label = DOCUMENT_CLASSES.index(image_path.parent.name)

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            bbox=encoding["bbox"].flatten(end_dim=1),
            pixel_values=encoding["pixel_values"].flatten(end_dim=1),
            labels=torch.tensor(label, dtype=torch.long),
        )


def prepare_dataset(dataset, directory, id2label):
    saved_dataset = dataset.map(
        lambda example, idx: process_and_save_image(example, idx, directory),
        with_indices=True,
        desc=f"Processing images for {directory}",
    )
    with open(f"{directory}/metadata.jsonl", "w") as outfile:
        for example in saved_dataset:
            write_obj = {
                "filepath": example["filepath"],
                "ground_truth": {"gt_parse": {"class": id2label[example["label"]]}},
            }
            outfile.write(json.dumps() + "\n")
