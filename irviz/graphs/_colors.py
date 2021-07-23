transparent_color_scales = {'TransparentRed': [[0, "rgba(255, 0, 0, 0)"],
                                               [1, "rgba(255, 0, 0, 255)"]],
                            'TransparentGreen': [[0, "rgba(0, 255, 0, 0)"],
                                                 [1, "rgba(0, 255, 0, 255)"]],
                            'TransparentBlue': [[0, "rgba(0, 0, 255, 0)"],
                                                [1, "rgba(0, 0, 255, 255)"]],
                            'TransparentYellow': [[0, "rgba(255, 255, 0, 0)"],
                                                  [1, "rgba(255, 255, 0, 255)"]],
                            'TransparentOrange': [[0, "rgba(255, 69, 0, 0)"],
                                                  [1, "rgba(255, 69, 0, 255)"]],
                            'TransparentPurple': [[0, "rgba(255, 0, 255, 0)"],
                                                  [1, "rgba(255, 0, 255, 255)"]],
                            'TransparentCyan': [[0, "rgba(0, 255, 255, 0)"],
                                                [1, "rgba(0, 255, 255, 255)"]]
                            }

decomposition_color_scales = ['gray'] + list(transparent_color_scales.keys())
