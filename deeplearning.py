def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    if device == 'gpu':
        model.to('cuda')

        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1

                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Loss: {:.4f}".format(running_loss/print_every))

                    running_loss = 0
                    
    else:
        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1


                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Loss: {:.4f}".format(running_loss/print_every))

                    running_loss = 0
    return model