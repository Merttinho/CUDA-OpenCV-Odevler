
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>

void sinDalgasi(std::vector<double>& sinyal, double faz, double periyot, double genlik) {

    for (int i = 0; i < sinyal.size(); i++) {
        sinyal[i] = genlik * std::sin((2 * CV_PI * i / periyot) + faz);
    }
}

void gurultuEkle(std::vector<double>& sinyal, double gurultuOrani, double genlik) {

    double gurultuGenligi = genlik * gurultuOrani;

    for (int i = 0; i < sinyal.size(); i++) {

        double rastgele = (double)rand() / RAND_MAX;

        double gurultu = (rastgele * 2.0 - 1.0) * gurultuGenligi;

        sinyal[i] += gurultu;
    }
}
void gaussFiltre(const std::vector<double>& giris,
    std::vector<double>& cikis) {

    double kernel[5] = { 1, 4, 6, 4, 1 };

    for (int i = 0; i < giris.size(); i++) {

        double toplam = 0.0;
        double agirlikToplami = 0.0;

        for (int k = -2; k <= 2; k++) {

            int indeks = i + k;

            if (indeks >= 0 && indeks < giris.size()) {

                double agirlik = kernel[k + 2];

                toplam += giris[indeks] * agirlik;
                agirlikToplami += agirlik;
            }
        }

        cikis[i] = toplam / agirlikToplami;
    }
}
void medyanFiltre(const std::vector<double>& giris,
    std::vector<double>& cikis) {

    for (int i = 0; i < giris.size(); i++) {

        std::vector<double> komsular;

        for (int k = -2; k <= 2; k++) {

            int indeks = i + k;

            if (indeks >= 0 && indeks < giris.size()) {
                komsular.push_back(giris[indeks]);
            }
        }

        for (int j = 0; j < komsular.size(); j++) {

            for (int m = j + 1; m < komsular.size(); m++) {

                if (komsular[j] > komsular[m]) {
                    double gecici = komsular[j];
                    komsular[j] = komsular[m];
                    komsular[m] = gecici;
                }
            }
        }

        cikis[i] = komsular[komsular.size() / 2];
    }
}


void grafikCiz(const std::vector<double>& sinyal,
    const std::string& pencereAdi) {

    cv::Mat grafik(500, 1024, CV_8UC3,
        cv::Scalar(255, 255, 255));

    cv::line(grafik,
        cv::Point(0, 250),
        cv::Point(1023, 250),
        cv::Scalar(0, 0, 0), 1);

    for (int i = 0; i < sinyal.size() - 1; i++) {

        int y1 = 250 - static_cast<int>(sinyal[i]);
        int y2 = 250 - static_cast<int>(sinyal[i + 1]);

        cv::line(grafik,
            cv::Point(i, y1),
            cv::Point(i + 1, y2),
            cv::Scalar(255, 0, 0), 1);
    }

    cv::imshow(pencereAdi, grafik);
}

int main() {

    srand(static_cast<unsigned int>(time(0)));

    std::vector<double> sinyal(1024);

    double faz = 0.0;
    double periyot = 128.0;
    double genlik = 100.0;

    double gurultuOrani = 0.30;

    sinDalgasi(sinyal, faz, periyot, genlik);

    std::vector<double> orijinalSinyal = sinyal;

    std::vector<double> gurultuluSinyal = orijinalSinyal;

    gurultuEkle(gurultuluSinyal, gurultuOrani, genlik);

    std::vector<double> gaussSonuc(1024);

    gaussFiltre(gurultuluSinyal, gaussSonuc);

    std::vector<double> medyanSonuc(1024);

    medyanFiltre(gurultuluSinyal, medyanSonuc);

    grafikCiz(orijinalSinyal, "1 - Orijinal Sinyal");

    grafikCiz(gurultuluSinyal, "2 - Gurultulu Sinyal");

    grafikCiz(gaussSonuc, "3 - Gauss Filtre");

    grafikCiz(medyanSonuc, "4 - Medyan Filtre");

    cv::waitKey(0);

    return 0;
}

