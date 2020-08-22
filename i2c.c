;;This C file has code to take data from pulse sensor and use ADS1115 to convert it into analog form using I2C protocol and to store the analog values in a CSV file. 


#include <stdio.h>
#include <stdlib.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <string.h>
#include <time.h>

void main()
{
        int i;
        int arr[7000];
        // Create I2C bus
        int file;
        char *bus = "/dev/i2c-0";
        if ((file = open(bus, O_RDWR)) < 0)
        {
                printf("Failed to open the bus. \n");
                exit(1);
 // Get I2C device, ADS1115 I2C address is 0x48(72)
         ioctl(file, I2C_SLAVE, 0x48);

        // Select configuration register(0x01)
        // AINP = AIN0 and AINN = AIN1, +/- 2.048V
        // Continuous conversion mode, 128 SPS(0x84, 0x83)
        char config[3] = {0};
        config[0] = 0x01;
        config[1] = 0x84;
        config[2] = 0x83;
        write(file, config, 3);
         sleep(1);
        for (i=0;i<6000;i++)
        {
        delay(20);
        // Read 2 bytes of data from register(0x00)
        // raw_adc msb, raw_adc lsb
        char reg[1] = {0x00};
        write(file, reg, 1);
	char data[2]={0};
 	if(read(file, data, 2) != 2)
        {
                printf("Error : Input/Output Error \n");
        }
        else
        {

                // Convert the data
                int raw_adc = (data[0] * 256 + data[1]);
                if (raw_adc > 32767)
                {
                         raw_adc -= 65535;
                }

                // Output data to screen
                printf("Digital Value of Analog Input: %d \n", raw_adc);
                arr[i]=raw_adc;

        }
	}
 for(i=0;i<6000;i++)
        {
                printf("%d \n",arr[i]);
        }
        FILE *filePointer ;
        filePointer = fopen("data1.csv", "w+");
        fprintf(filePointer, "Raw_PPG_Green\n");
        for (i=0;i<6000;i++)
        {
                fprintf(filePointer, "%d \n", arr[i]);
        }
        fclose(filePointer);
        return 0;


}






