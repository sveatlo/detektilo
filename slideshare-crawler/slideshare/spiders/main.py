# -*- coding: utf-8 -*-
import scrapy


class MainSpider(scrapy.Spider):
    name = 'main'
    allowed_domains = ['slideshare.net']
    start_urls = ['https://slideshare.net/']

    def parse(self, response):
        for slideshow_url in response.css('.iso_slideshow_link').xpath('@href').extract():
            yield response.follow(response.urljoin(slideshow_url), self.parse_slideshow)

    def parse_slideshow(self, response):
        yield {
            'image_urls': response.css('.slide_image').xpath('@data-full').extract(),
            'images': [],
        }

        for slideshow_url in response.css('.slideview_related_item').xpath('@href').extract():
            yield response.follow(response.urljoin(slideshow_url), self.parse_slideshow)
