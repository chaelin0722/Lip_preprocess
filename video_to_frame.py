import os
### video to frames  (LRW)
dir = "/???/"
output_dir = "/??/"

folders = os.listdir(dir)
folders.sort()
for folder in folders:

    if not os.path.exists(os.path.join(output_dir +  folder)):
        os.mkdir(os.path.join(output_dir + folder))


    for sub in ['train', 'val', 'test']:

        if not os.path.exists(os.path.join(dir + folder + "/" + sub)):
            print("no folder exists : " , sub)
            continue

        if not os.path.exists(os.path.join(output_dir + folder + "/" + sub)):
            os.mkdir(os.path.join(output_dir + folder + "/" + sub))

        file_dir = os.path.join(dir + folder + "/" + sub)
        out_dir = os.path.join(output_dir + folder + "/" + sub)

        mp4_files = [mp4 for mp4 in os.listdir(file_dir) if mp4.endswith('.mp4')]

        for file in mp4_files:
            if not os.path.exists(os.path.join(out_dir + "/" + str(os.path.splitext(file)[0]))):
                os.mkdir(os.path.join(out_dir+ "/" + str(os.path.splitext(file)[0])))

            files_dir = os.path.join(file_dir + "/" + str(os.path.splitext(file)[0]))
            outs_dir = os.path.join(out_dir+ "/" + str(os.path.splitext(file)[0]))

            print("working on", files_dir)
            #if len(os.listdir(files_dir)) == len(os.listdir(outs_dir)):
            #    print("same")
            #    continue

            command = "ffmpeg -r 1 -i " + file_dir + "/"  + str(file) + " -r 1 " + outs_dir + "/%03d.png"
            os.system(command=command)
            print("done one mp4 file")

        # video to frames
        #command = "ffmpeg -r 1 -i /FER/AFEW/Val_AFEW/" + sub + "/" + str(filename) + " -r 1 '/AFEW/PREPROCESSED_AFEW/VALID/frames/" + emotion + "/" + str(os.path.splitext(filename)[0]) + "/%03d.png'"
        #os.system(command=command)
