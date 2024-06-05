import torch


def matchTemplateTorch(img_tensor, template_tensor):
    img_tensor = img_tensor.clone()
    template_tensor = template_tensor.clone()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    template_tensor = template_tensor.unsqueeze(1)
    img_tensor[img_tensor == 0] = -1
    template_tensor[template_tensor == 0] = -1
    result1 = torch.nn.functional.conv2d(
        img_tensor, template_tensor, bias=None, stride=1, padding=0
    )
    # -1 not equal and 1 match
    max_value = template_tensor.size(2) * template_tensor.size(3)
    result1 = (result1 + max_value) / (max_value * 2)
    return result1.squeeze(0)


def get_top_char_pairs(res, chars):
    all_results = []
    for r, c in zip(res, chars):
        y, x = torch.unravel_index(r.argmax(), r.shape)
        all_results.append(
            {"char": c, "score": r[y, x].item(), "y": y.item(), "x": x.item()}
        )
    all_results.sort(key=lambda x: x["score"], reverse=True)

    grouped = {}
    for item in all_results:
        added = False
        for g in grouped.keys():
            if abs(g - item["y"]) < 10:
                grouped[g].append(item)
                added = True
                break
        if not added:
            grouped[item["y"]] = [item]

    # check every combination of 2
    valid_combinations = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        # check every combination of 2
        for i in range(len(group)):
            for j in range(len(group)):

                if (group[i]["x"] - group[j]["x"]) > 90 and (
                    group[i]["x"] - group[j]["x"]
                ) < 150:
                    valid_combinations.append(
                        {
                            "text": group[j]["char"] + group[i]["char"],
                            "score": group[j]["score"] * group[i]["score"],
                        }
                    )
    valid_combinations.sort(key=lambda x: x["score"], reverse=True)
    return valid_combinations
